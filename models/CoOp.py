import os.path as osp

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from open_clip.tokenizer import SimpleTokenizer,tokenize


class TextEncoder(nn.Module):
    def __init__(self, clip_model):

        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_,_ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



class PromptLearner(nn.Module):
    def __init__(self,
                 prompts,
                 n_ctx, # prompt max len
                 CSC, # True or False multi prompt
                 class_token_position, # cls position
                 clip_model):

        super().__init__()

        ctx_dim = clip_model.ln_final.weight.shape[0] #

        self.ctx={}

        for cls in prompts:
            for position in class_token_position:
                if CSC:
                    ctx_vectors = torch.empty(len(prompts[cls]), n_ctx, ctx_dim).to(clip_model.device)
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim).to(clip_model.device)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx['{}_{}'.format(cls,position)]=nn.Parameter(ctx_vectors,requires_grad=True)

        self.ctx = nn.ParameterDict(self.ctx)  # to be optimized

        prompt_prefix = " ".join(["X"] * n_ctx)

        _tokenizer = SimpleTokenizer()

        prompts_split={cls: [prompt.replace("_", " ")  for prompt in prompts[cls]] for cls in prompts}

        prompts_lens= {cls: [ len(_tokenizer.encode(prompt)) for prompt in prompts_split[cls]] for cls in prompts_split}

        prompts_learnable_tokens = {cls:[prompt_prefix + " " + prompt + "." for prompt in prompts_split[cls]] for cls in prompts_split}

        tokenized_prompts = {cls:torch.cat([tokenize(prompt) for prompt in prompts_learnable_tokens[cls]]).to(clip_model.device) for cls in prompts_learnable_tokens}

        with torch.no_grad():
            embeddings = {cls:clip_model.token_embedding(tokenized_prompts[cls])  for cls in tokenized_prompts}

        self.register_embeddings={}

        for cls in embeddings:
            self.register_embeddings['{}_token_prefix'.format(cls)]=embeddings[cls][:, :1, :]
            self.register_embeddings['{}_token_suffix'.format(cls)]=embeddings[cls][:, 1 + n_ctx :, :]

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.prompts_lens = prompts_lens
        self.class_token_position = class_token_position


    def forward(self):
        cls_prompts={}

        for cls in self.tokenized_prompts:

            prefix =  self.register_embeddings['{}_token_prefix'.format(cls)]
            suffix =  self.register_embeddings['{}_token_suffix'.format(cls)]

            cls_prompts[cls]=[]

            for position in self.class_token_position:

                ctx = self.ctx['{}_{}'.format(cls,position)]
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(len(self.prompts_lens[cls]), -1, -1)

                if position == "end":
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,     # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )

                elif position == "middle":

                    half_n_ctx = self.n_ctx // 2
                    prompts = []

                    for i in range(len(self.prompts_lens[cls])):
                        p_len = self.prompts_lens[cls][i]

                        prefix_i = prefix[i : i + 1, :, :]
                        class_i = suffix[i : i + 1, :p_len, :]
                        suffix_i = suffix[i : i + 1, p_len:, :]
                        ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                        ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]

                        prompt = torch.cat(
                            [
                                prefix_i,     # (1, 1, dim)
                                ctx_i_half1,  # (1, n_ctx//2, dim)
                                class_i,      # (1, name_len, dim)
                                ctx_i_half2,  # (1, n_ctx//2, dim)
                                suffix_i,     # (1, *, dim)
                            ],
                            dim=1,
                        )
                        prompts.append(prompt)
                    prompts = torch.cat(prompts, dim=0)

                else :
                    assert position == "front"
                    prompts = []

                    for i in range(len(self.prompts_lens[cls])):
                        p_len = self.prompts_lens[cls][i]

                        prefix_i = prefix[i : i + 1, :, :]
                        class_i = suffix[i : i + 1, :p_len, :]
                        suffix_i = suffix[i : i + 1, p_len:, :]
                        ctx_i = ctx[i : i + 1, :, :]
                        prompt = torch.cat(
                            [
                                prefix_i,  # (1, 1, dim)
                                class_i,   # (1, name_len, dim)
                                ctx_i,     # (1, n_ctx, dim)
                                suffix_i,  # (1, *, dim)
                            ],
                            dim=1,
                        )
                        prompts.append(prompt)

                    prompts = torch.cat(prompts, dim=0)

                cls_prompts[cls].append(prompts)
            cls_prompts[cls]=torch.cat(cls_prompts[cls],dim=0)
        return cls_prompts


class PromptMaker(nn.Module):

    def __init__(self,
                 prompts,
                 clip_model,
                 n_ctx: int=8,  # prompt max len
                 CSC: bool= True,  # True or False multi prompt
                 class_token_position: list=['end'],  # cls position
                 ):

        super().__init__()
        assert 'normal' in prompts and 'abnormal' in prompts

        for position in class_token_position:
            assert  position in ['end','middle','front']

        self.prompt_learner = PromptLearner(prompts, n_ctx, CSC, class_token_position, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.class_token_position = class_token_position
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, image_features):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features=[]

        for cls in prompts:
            class_embedding = self.text_encoder(prompts[cls], tokenized_prompts[cls].repeat(len(self.class_token_position),1))
            class_embedding = class_embedding.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1)
        return text_features
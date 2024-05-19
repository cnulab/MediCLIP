import open_clip
import torch
import yaml
from easydict import EasyDict
from models.Necker import Necker
from models.Adapter import Adapter
import math
import argparse
import warnings
from utils.misc_helper import *
from torch.utils.data import DataLoader
from models.MapMaker import MapMaker
from utils.losses import FocalLoss,BinaryDiceLoss
from datasets.dataset import TrainDataset,\
                                ChexpertTestDataset,\
                                BusiTestDataset,\
                                BrainMRITestDataset
import pprint
from tqdm import tqdm
warnings.filterwarnings('ignore')


@torch.no_grad()
def make_vision_takens_info(model,model_cfg,layers_out):

    img = torch.ones((1,3,model_cfg['vision_cfg']['image_size'],
                          model_cfg['vision_cfg']['image_size'])).to(model.device)

    img_feature,tokens = model.encode_image(img,layers_out)

    if len(tokens[0].shape)==3:
        model.token_size= [int(math.sqrt(token.shape[1]-1)) for token in tokens]
        model.token_c= [token.shape[-1]  for token in tokens]
    else:
        model.token_size = [token.shape[2] for token in tokens]
        model.token_c = [token.shape[1] for token in tokens]

    model.embed_dim = model_cfg['embed_dim']
    print("model token size is {}".format(model.token_size)," model token dim is {}".format(model.token_c))


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(args.config.model_name, args.config.image_size, device=device)

    for param in model.parameters():
        param.requires_grad_(False)

    args.config.model_cfg = model_cfg

    make_vision_takens_info(model,
                            args.config.model_cfg,
                            args.config.layers_out)

    current_time = get_current_time()
    args.config.save_root=os.path.join(args.config.save_root,current_time)

    if not os.path.exists(args.config.save_root):
        os.makedirs(args.config.save_root)

    logger = create_logger("logger",os.path.join(args.config.save_root,'logger.log'))
    logger.info("config: {}".format(pprint.pformat(args)))

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model,target=args.config.model_cfg['embed_dim']).to(model.device)

    if args.config.prompt_maker=='coop':
        from models.CoOp import PromptMaker
        logger.info("load CoOp")
    else:
        raise NotImplementedError("type of prompt must in ['coop']")

    prompt_maker = PromptMaker(
        prompts=args.config.prompts,
        clip_model=model,
        n_ctx= args.config.n_learnable_token,
        CSC = args.config.CSC,
        class_token_position=args.config.class_token_positions,
    ).to(model.device)

    map_maker = MapMaker(image_size=args.config.image_size).to(model.device)

    optimizer = torch.optim.Adam([
            {'params': prompt_maker.prompt_learner.parameters(),'lr': 0.001},
            {'params': adapter.parameters(),"lr":0.001},
        ], lr=0.001, betas=(0.5, 0.999))

    train_dataset = TrainDataset(args=args.config,
                                    source=os.path.join(args.config.data_root,args.config.train_dataset),
                                    preprocess=preprocess,
                                    k_shot=args.k_shot)

    train_dataloader = DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True, num_workers=2)

    test_dataloaders = {}
    best_record = {}

    for test_dataset_name in args.config.test_datasets:

        if test_dataset_name == 'chexpert':
            test_dataset = ChexpertTestDataset( args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess,
                                            )

        elif test_dataset_name =='brainmri':

            test_dataset = BrainMRITestDataset(
                                            args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess,
                                            )
        elif test_dataset_name =='busi':

            test_dataset = BusiTestDataset(
                                            args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess)
        else:
            raise NotImplementedError("dataset must in ['chexpert','busi','brainmri'] ")

        test_dataloader = DataLoader(test_dataset, batch_size=args.config.batch_size,num_workers=2)
        test_dataloaders[test_dataset_name]=test_dataloader
        best_record[test_dataset_name]=None

    logger.info("train data ({}) len {}".format(args.config.train_dataset,len(train_dataset)))

    for test_dataset_name in test_dataloaders:
        logger.info("test data ({}) len {}".format(test_dataset_name, len(test_dataloaders[test_dataset_name].dataset)))

    for task_name in args.config.anomaly_tasks:
        logger.info("anomaly syn task is {}, sampling probability is {}".format(task_name,args.config.anomaly_tasks[task_name]))

    for epoch in range(0, args.config.epoch):
        last_iter = epoch * len(train_dataloader)

        train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            last_iter,
            logger,
            model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
        )

        if (epoch+1) % args.config.val_freq_epoch == 0:

            results = validate(args,test_dataloaders, epoch,model, necker,adapter,prompt_maker,map_maker)
            save_flag = False

            for test_dataset_name in results:
                if best_record[test_dataset_name] is None:
                    if test_dataset_name=='busi':
                        best_record[test_dataset_name] = [results[test_dataset_name]["image-auroc"],
                                                          results[test_dataset_name]['pixel-auroc']]
                    else:
                        best_record[test_dataset_name] = [results[test_dataset_name]["image-auroc"]]

                    save_flag=True
                else:
                    if np.mean([results[test_dataset_name][key] for key in results[test_dataset_name]]) > np.mean(best_record[test_dataset_name]):
                        if test_dataset_name == 'busi':
                            best_record[test_dataset_name] = [results[test_dataset_name]["image-auroc"],
                                                              results[test_dataset_name]['pixel-auroc']]
                        else:
                            best_record[test_dataset_name] = [results[test_dataset_name]["image-auroc"]]
                        save_flag=True


                if test_dataset_name=='busi':
                    logger.info("({}): Epoch: {}, image auroc: {:.4f}, pixel_auroc: {:.4f},".format(test_dataset_name,
                                                                                                    epoch+1,
                                                                                                    results[test_dataset_name]["image-auroc"],
                                                                                                    results[test_dataset_name]['pixel-auroc']))
                else:
                    logger.info("({}): Epoch: {}, image auroc: {:.4f},".format(
                        test_dataset_name,
                        epoch+1,
                        results[test_dataset_name]["image-auroc"],
                    ))

            for test_dataset_name in results:
                if test_dataset_name == 'busi':
                    logger.info(
                        "({} best): image auroc: {:.4f}, pixel auroc: {:.4f},".format(
                            test_dataset_name,
                            best_record[test_dataset_name][0],
                            best_record[test_dataset_name][1],
                        ))
                else:
                    logger.info(
                        "({} best): image auroc: {:.4f},".format(
                            test_dataset_name,
                            best_record[test_dataset_name][0],
                        ))

            if save_flag:
                logger.info("save checkpoints in epoch: {}".format(epoch+1))
                torch.save({
                        "adapter_state_dict": adapter.state_dict(),
                        "prompt_state_dict": prompt_maker.prompt_learner.state_dict(),
                    }, os.path.join(args.config.save_root, 'checkpoints_{}.pkl'.format(epoch + 1)))


def train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            start_iter,
            logger,
            clip_model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
):

    loss_meter = AverageMeter(args.config.print_freq_step)

    focal_criterion = FocalLoss()
    dice_criterion = BinaryDiceLoss()

    adapter.train()
    prompt_maker.train()

    for i, input in enumerate(train_dataloader):
        curr_step = start_iter + i

        images = input['image'].to(clip_model.device)
        gt_mask = input['mask'].to(clip_model.device)

        with torch.no_grad():
            _, image_tokens = clip_model.encode_image(images,out_layers=args.config.layers_out)
            image_features = necker(image_tokens)

        vision_adapter_features = adapter(image_features)
        propmt_adapter_features = prompt_maker(vision_adapter_features)
        anomaly_map = map_maker(vision_adapter_features,propmt_adapter_features)

        loss = []

        loss.append(focal_criterion(anomaly_map,gt_mask))
        loss.append(dice_criterion(anomaly_map[:, 1, :, :],gt_mask))

        loss = torch.sum(torch.stack(loss))
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (curr_step + 1) % args.config.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                    .format(
                    epoch+1 ,
                    args.config.epoch,
                    curr_step + 1,
                    len(train_dataloader) * args.config.epoch,
                    loss=loss_meter,
                )
            )


def validate(args, test_dataloaders, epoch, clip_model, necker, adapter, prompt_maker, map_maker):

    adapter.eval()
    prompt_maker.eval()
    results = {}

    for test_dataset_name in test_dataloaders:
        test_dataloader = test_dataloaders[test_dataset_name]

        anomaly_maps = []
        anomaly_gts = []

        image_scores = []
        image_labels = []

        with torch.no_grad():
            for i, input in enumerate(tqdm(test_dataloader,desc=test_dataset_name)):

                images = input['image'].to(clip_model.device)

                _, image_tokens = clip_model.encode_image(images, out_layers=args.config.layers_out)
                image_features = necker(image_tokens)
                vision_adapter_features = adapter(image_features)
                propmt_adapter_features = prompt_maker(vision_adapter_features)
                anomaly_map = map_maker(vision_adapter_features, propmt_adapter_features)

                B,_,H,W = anomaly_map.shape

                anomaly_map = anomaly_map[:,1,:,:]
                anomaly_gt = input['mask']

                anomaly_maps.append(anomaly_map.cpu().numpy())
                anomaly_gts.append(anomaly_gt.cpu().numpy())

                anomaly_score,_ = torch.max(anomaly_map.view((B,H*W)), dim=-1)

                image_scores.extend(anomaly_score.cpu().numpy().tolist())
                image_labels.extend(input['is_anomaly'].cpu().numpy().tolist())

        metric = compute_imagewise_metrics(image_scores,image_labels)

        if test_dataset_name=='busi':
            metric.update(compute_pixelwise_metrics(anomaly_maps,anomaly_gts))

        results[test_dataset_name] = metric
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MediCLIP")
    parser.add_argument("--config_path", type=str, default='config/brainmri.yaml', help="model configs")
    parser.add_argument("--k_shot", type=int, default=16, help="normal image number")
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

#关闭cudnn
torch.backends.cudnn.enabled = False
# from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import lib.utils as utils
from IMTS.lib.data_process.parse_datasets import parse_datasets
from lib.models.tPatchGNN import *
from lib.models.visionts import VisionTS
from IMTS.lib.models.mean_model import MeanModel

parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

# parser.add_argument('--save_dir', type=str, default='pretrain_weights/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")
parser.add_argument('--log_dir', type=str, default=None, help="e.g. logs/ushcn/log_dir")
parser.add_argument('--pretrain_weights_dir', type=str, default=None, help="dir for pretrained weights")


# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='VisionTS-B', help="Model name [tPatchGNN, VisionTS-B, VisionTS-L, VisionTS-H, MeanModel]")
parser.add_argument('--mask_flag', action='store_true', help="whether to use prex attn mask")
# parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')
parser.add_argument('--log_suffix', type=str, default='', help='suffix discription on log file name for this run')
parser.add_argument('--periodicity', type=int, default=None, help='periodicity for datasets')
parser.add_argument('--mask_ratio', type=float, default=None, help='mask ratio for self-supervised learning')
parser.add_argument('--train_mode', type=str, default='pre-train', help='pre-train/fine-tune')

# Test-Time-Training configs
parser.add_argument('--TTT', action='store_true', help='whether to use TTT')
parser.add_argument('--ttt_iter', type=int, default=None, help='number of TTT iterations')
parser.add_argument('--ttt_lr', type=float, default=None, help='learning rate for TTT')

# lora
parser.add_argument('--apply_lora', action='store_true', help='whether to apply lora')
parser.add_argument('--lora_r', type=int, default=8, help='lora rank')
parser.add_argument('--lora_alpha', type=int, default=1, help='lora alpha')
parser.add_argument('--lora_dropout', type=float, default=0., help='lora dropout')
parser.add_argument('--merge_weights', action='store_true', help='whether to merge weights')
parser.add_argument('--finetune_type', type=str, default='freeze', help='finetune type: norm, bias, none, mlp, attn')
parser.add_argument('--encoder_only', action='store_true', help='whether to use encoder-extracted repesentations to predict')

# 训练多少百分比的数据
parser.add_argument('--few_shot_ratio', type=float, default=None, help='train percent')
parser.add_argument('--no_vision_pre', action='store_true', help='whether to use vision pretrain')
parser.add_argument('--wo_gcn', action='store_true', help='whether to use gcn')

# parser.add_argument('--ttt_mode', type=str, default='offline', help='online or offline')

# parser.add_argument('--noise_std', type=float, default=0.0, help='noise for data augmentation')

args = parser.parse_args()

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()  # 确保所有进程都已初始化

def cleanup():
    dist.destroy_process_group()


# number of history patches
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]


#####################################################################################################

def main(args):
    
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	### loadding model specific config ###
	with open(f"./configs/{args.model}.yaml") as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
			# merge model-specific args and config
			for key in config:
				if key not in args.__dict__:
					args.__dict__[key] = config[key]

	if args.no_vision_pre:
		args.vm_pretrained = False
		args.vm_ckpt = None

	### data setting ###
	data_obj = parse_datasets(args, patch_ts=(args.if_patch)) # 仅tpatchGNN使用patch法处理数据集
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	
	####
	

	### initialize model ###
	if args.model == "tPatchGNN":
		model = tPatchGNN(args).to(args.device)
	elif "VisionTS" in args.model:
		model = VisionTS(args=args,arch=args.arch, finetune_type=args.ft_type, load_ckpt=args.vm_pretrained == 1, ckpt_dir=args.vm_ckpt).to(args.device)
		if args.train_mode == 'fine-tune' and args.pretrain_weights_dir != None:
				model = utils.get_ckpt_model(args.pretrain_weights_dir, model, args.device)
	elif args.model == "MeanModel":
		model = MeanModel().to(args.device)

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}/_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = f"logs/{args.dataset}"+f"{('/'+ args.log_dir) if not args.log_dir is None else ''}/{args.model}_{args.hid_dim}hid_dim_{args.mask_ratio}mask_ratio_{args.patch_size}patch_{args.stride}stride_{args.nlayer}layer_{args.lr}lr_{args.seed}seed_{args.batch_size}bs_layernorm_{args.arch}_{args.hid_dim}hidden_dim_{args.te_dim}te_dim_{args.node_dim}node_dim.log"
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	if not os.path.exists("logs/{}".format(args.dataset)):
		utils.makedirs("logs/{}".format(args.dataset))
	if (args.log_dir!=None) and (not os.path.exists(f"logs/{args.dataset}/{args.log_dir}")):
		utils.makedirs(f"logs/{args.dataset}/{args.log_dir}")
	if not os.path.exists(f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/") and args.train_mode == 'pre-train':
		utils.makedirs(f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/")
	# if not os.path.exists(f"logs/{args.dataset}/{args.log_dir}/ft_weights/") and args.train_mode == 'fine-tune':
	# 	utils.makedirs(f"logs/{args.dataset}/{args.log_dir}/ft_weights/")
		
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)
 

	##################################################################

	if args.model == "MeanModel":
		test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])	
		logger.info('ExpID {}'.format(experimentID))
		if(test_res != None):
			logger.info("Test Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format( test_res["loss"], test_res["mse"],\
					test_res["rmse"], test_res["mae"], test_res["mape"]*100))
		exit(0)

	##################################################################
	if args.train_mode == 'fine-tune':
		finetune_model_params = []
		for name, param in model.named_parameters():
			if (args.apply_lora and 'lora' in name) or 'vision_model' not in name:
				param.requires_grad = True
				finetune_model_params.append(param)
			###
			elif args.finetune_type == 'wo_ssl_ft' and (('vision_model' in name and 'block' not in name and 'decoder' not in name) or 'norm' in name):
				# print_list.append(name)
				param.requires_grad = True
				finetune_model_params.append(param)
			##
			elif args.finetune_type == 'norm':
				if 'norm' in name and 'vision_model' in name:
					param.requires_grad = True
					finetune_model_params.append(param)
			elif args.finetune_type == 'attn':
				if 'attn' in name and 'vision_model' in name:
					param.requires_grad = True
					finetune_model_params.append(param)
			elif args.finetune_type == 'mlp':
				if 'mlp' in name and 'vision_model' in name:
					param.requires_grad = True
					finetune_model_params.append(param)
			elif args.finetune_type == 'bias':
				if 'bias' in name and 'vision_model' in name:
					param.requires_grad = True
					finetune_model_params.append(param)
			elif args.finetune_type == 'all':
				param.requires_grad = True
				finetune_model_params.append(param)
			else:
				param.requires_grad = False
		optimizer = optim.Adam(finetune_model_params, lr=args.lr, weight_decay=args.w_decay)
		# 添加学习率调度器
		# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.01)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

	# def get_parameter_number(model):
	# 	total_num = sum(p.numel() for p in model.parameters())
	# 	trainable_num = sum(p.numel() for p in finetune_model_params)
	# 	return {'Total': total_num, 'Trainable': trainable_num}
	# print(get_parameter_number(model))

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	# best_val_mae = np.inf
	best_val_mse = np.inf
 
	test_res = None
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		model.train()
		# for _ in tqdm(range(num_batches)):
		# 更新vision_model以外的参数
		with tqdm(total=num_batches) as tq:
			for _ in range(num_batches):
					optimizer.zero_grad()
					batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
					train_res = compute_all_losses(model, batch_dict)
					train_res["loss"].backward()
					# 显示训练过程中的epoch, loss tqdm
					tq.set_description('Epoch %i' % itr)
					tq.set_postfix(loss=train_res["loss"].item())
					tq.update(1)
					optimizer.step()

		# ### Validation ###
		# if args.train_mode == 'fine-tune':
		# 	model.eval()
		# 	with torch.no_grad():
		# 		val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
		# 	if val_res["mse"] < best_val_mse:
		# 		best_val_mse = val_res["mse"]
		# 		best_iter = itr
				
		# 		# 保存当前模型参数
		# 		model_state = {k: v.clone() for k, v in model.state_dict().items()}
	
		# 		# 加载extractor_params
		# 		for name, param in model.named_parameters():
		# 			if name in extractor_params:
		# 				param.data = model_state[name].data
				
		# 		# TTT阶段
		# 		model.train_mode_update('ttt')  # 切换到TTT模式
		# 		model.train()
				
		# 		# 只优化vision_model的参数
		# 		ttt_params = []
		# 		for name, param in model.named_parameters():
		# 			if 'vision_model' in name:
		# 				param.requires_grad = True
		# 				ttt_params.append(param)
		# 			else:
		# 				param.requires_grad = False
				
		# 		test_optimizer = optim.Adam(ttt_params, lr=args.ttt_lr)
    
				
		# 		# TTT训练
		# 		for _ in tqdm(range(args.ttt_iter)):
		# 			test_optimizer.zero_grad()
		# 			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
		# 			test_train_res = compute_all_losses(model, batch_dict)
		# 			test_train_res["loss"].backward()
		# 			test_optimizer.step()
				
		# 		# 最终评估
		# 		model.train_mode_update("fine-tune")
				
		# 		# 恢复vision_model以外的参数
		# 		for name, param in model.named_parameters():
		# 			if name in extractor_params.keys():
		# 				param.data = model_state[name].data
    
		# 		model.eval()
		# 		with torch.no_grad():
		# 			test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
				
		# 		# 恢复原始模型参数
		# 		model.load_state_dict(model_state)
			
		# 	logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
		# 	logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
		# 	logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
		# 		.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
		# 	if test_res != None:
		# 		logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
		# 			.format(best_iter, test_res["loss"], test_res["mse"],\
		# 			test_res["rmse"], test_res["mae"], test_res["mape"]*100))
		# 	logger.info("Time spent: {:.2f}s".format(time.time()-st))

  
		# else:
		model.eval()
		with torch.no_grad():
			# val_start_time = time.time()
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			# val_end_time = time.time()
			# print("Validation time: {:.2f}s".format(val_end_time - val_start_time))
			
			### Testing ###
			# if(val_res["mae"] < best_val_mae):
			if(val_res["mse"] < best_val_mse):
				# best_val_mae = val_res["mae"]
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
				if args.train_mode == 'pre-train':
					torch.save(model.state_dict(), f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/Exp{experimentID}_{args.model}_hid_dim{args.hid_dim}_te_dim{args.te_dim}_node_dim{args.node_dim}_mask_ratio{args.mask_ratio}_stride{args.stride}_patchsize{args.patch_size}_seed{args.seed}_nlayer{args.nlayer}_best.pth")
				# if args.train_mode == 'fine-tune':
				# 	torch.save(model.state_dict(), f"logs/{args.dataset}/{args.log_dir}/ft_weights/Exp{experimentID}_{args.model}_hid_dim{args.hid_dim}_te_dim{args.te_dim}_node_dim{args.node_dim}_mask_ratio{args.mask_ratio}_stride{args.stride}_patchsize{args.patch_size}_seed{args.seed}_nlayer{args.nlayer}_best.pth")
			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
					test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))

  
		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			sys.exit(0)

		# # 在训练循环末尾添加scheduler.step()
		# if args.train_mode == 'fine-tune':
		# 	scheduler.step()


def ddp_main(rank, world_size, args):
    
	args.world_size = world_size
    
	setup(rank, world_size)
    
    # 设置随机种子确保所有进程具有相同的数据分布等
	utils.setup_seed(args.seed + rank)

    # 更新设备信息
	args.device = torch.device(f"cuda:{rank}")

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)

	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	### loadding model specific config ###
	with open(f"./configs/{args.model}.yaml") as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
			# merge model-specific args and config
			for key in config:
				if key not in args.__dict__:
					args.__dict__[key] = config[key]

	### data setting ###
	data_obj = parse_datasets(args, patch_ts=(args.if_patch), rank=rank) # 仅tpatchGNN使用patch法处理数据集
	input_dim = data_obj["input_dim"]
	train_sampler = data_obj["train_sampler"]

 
	### Model setting ###
	args.ndim = input_dim
	
	####
	

	### initialize model ###
	if args.model == "tPatchGNN":
		model = tPatchGNN(args).to(args.device)
	elif "VisionTS" in args.model:
		model = VisionTS(args=args,arch=args.arch, finetune_type=args.ft_type, load_ckpt=args.vm_pretrained == 1, ckpt_dir=args.vm_ckpt).to(args.device)
		if args.train_mode == 'fine-tune' and args.pretrain_weights_dir != None:
				model = utils.get_ckpt_model(args.pretrain_weights_dir, model, args.device)
	elif args.model == "MeanModel":
		model = MeanModel().to(args.device)
  
	# 使用 DDP 包装模型
	model = DDP(model, device_ids=[rank])

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}/_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = f"logs/{args.dataset}"+f"{('/'+ args.log_dir) if not args.log_dir is None else ''}/{args.model}_{args.mask_ratio}mask_ratio_{args.periodicity}periodicity_{args.patch_size}patch_{args.stride}stride_{args.nlayer}layer_{args.lr}lr_{args.seed}seed_{args.batch_size}bs_layernorm_{args.arch}_lora_r{args.lora_r}_lora_alpha{args.lora_alpha}_lora_dropout{args.lora_dropout}.log"
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	if not os.path.exists("logs/{}".format(args.dataset)):
		utils.makedirs("logs/{}".format(args.dataset))
	if (args.log_dir!=None) and (not os.path.exists(f"logs/{args.dataset}/{args.log_dir}")):
		utils.makedirs(f"logs/{args.dataset}/{args.log_dir}")
	if not os.path.exists(f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/") and args.train_mode == 'pre-train':
		utils.makedirs(f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/")
	
	if rank == 0:
		logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		logger.info(input_command)
		logger.info(args)
 

	##################################################################

	if args.model == "MeanModel":
		test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])	
		logger.info('ExpID {}'.format(experimentID))
		if(test_res != None):
			logger.info("Test Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format( test_res["loss"], test_res["mse"],\
					test_res["rmse"], test_res["mae"], test_res["mape"]*100))
		exit(0)

	##################################################################

	if args.train_mode == 'fine-tune':
		finetune_model_params = []
		for name, param in model.named_parameters():
			if (args.apply_lora and 'lora' in name) or 'vision_model' not in name:
				param.requires_grad = True
				finetune_model_params.append(param)
			else:
				param.requires_grad = False
		optimizer = optim.Adam(finetune_model_params, lr=args.lr, weight_decay=args.w_decay)
		# 添加学习率调度器
		# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.01)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	if rank == 0:	
		print("n_train_batches:", num_batches)

	# best_val_mae = np.inf
	best_val_mse = np.inf
 
	test_res = None
	for itr in range(args.epoch):
		train_sampler.set_epoch(itr)
  
		dist.barrier()
  
		st = time.time()

		### Training ###
		model.train()
		# for _ in tqdm(range(num_batches)):
		with tqdm(total=num_batches) as tq:
			for _ in range(num_batches):
					optimizer.zero_grad()
					batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
					train_res = compute_all_losses(model, batch_dict)
					train_res["loss"].backward()
					# 显示训练过程中的epoch, loss tqdm
					tq.set_description('Epoch %i' % itr)
					tq.set_postfix(loss=train_res["loss"].item())
					tq.update(1)
					optimizer.step()

		# 每个epoch结束后设置同步点
		dist.barrier()
        
		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			# if(val_res["mae"] < best_val_mae):
			if(val_res["mse"] < best_val_mse):
				# best_val_mae = val_res["mae"]
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
				if args.train_mode == 'pre-train' and rank == 0:
					torch.save(model.state_dict(), f"logs/{args.dataset}/{args.log_dir}/pretrain_weights/Exp{experimentID}_{args.model}_mask_ratio{args.mask_ratio}_stride{args.stride}_patchsize{args.patch_size}_seed{args.seed}_nlayer{args.nlayer}_best.pth")
			
			if rank == 0:
				logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
				logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
				logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
				if(test_res != None):
					logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
						.format(best_iter, test_res["loss"], test_res["mse"],\
						test_res["rmse"], test_res["mae"], test_res["mape"]*100))
				logger.info("Time spent: {:.2f}s".format(time.time()-st))

		dist.barrier()

		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			cleanup()
			sys.exit(0)


if __name__ == '__main__':
	world_size = torch.cuda.device_count()
	args.world_size = world_size
	if world_size == 1:
		main(args)
	if world_size > 1:
		torch.multiprocessing.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)
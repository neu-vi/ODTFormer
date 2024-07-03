from .experiment import *
from .wandb_utils import *

import torch.distributed as dist
import torch.nn as nn
import functools

def find_free_port():
	import socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# Binding to port 0 will cause the OS to find an available port for us
	sock.bind(("", 0))
	port = sock.getsockname()[1]
	sock.close()
	# NOTE: there is still a chance the port could be taken by other processes.
	return port

def get_world_size() -> int:
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()


def get_rank() -> int:
	if not dist.is_available():
		return 0
	if not dist.is_initialized():
		return 0
	return dist.get_rank()


def get_local_rank() -> int:
	"""
	Returns:
		The rank of the current process within the local (per-machine) process group.
	"""
	if not dist.is_available():
		return 0
	if not dist.is_initialized():
		return 0
	assert _LOCAL_PROCESS_GROUP is not None
	return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
	"""
	Returns:
		The size of the per-machine process group,
		i.e. the number of processes per machine.
	"""
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
	return get_rank() == 0


def synchronize():
	"""
	Helper function to synchronize (barrier) among all processes when
	using distributed training
	"""
	if not dist.is_available():
		return
	if not dist.is_initialized():
		return
	world_size = dist.get_world_size()
	if world_size == 1:
		return
	dist.barrier()
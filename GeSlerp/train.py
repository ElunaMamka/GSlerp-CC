# from rex.tasks.base_task import TaskBase
from rex.tasks.base_task import TaskBase
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.logging import logger
from rex.utils.registry import get_registered, import_module_and_submodules, register

import torch
# import os
torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

@register("rex_init_call")
def train(cmd_args=None):
    parser, args = ConfigParser.parse_cmd_args(
        ConfigArgument(
            "-m",
            "--include-package",
            action="append",
            type=str,
            default=[],
            help="packages to load",
        ),
        cmd_args=cmd_args,
    )
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)
    config = ConfigParser.parse_args_config(parser, args)
    if "task_type" not in config:
        raise ValueError(
            (
                "`task_type` must exist in config! "
                "If you are using the default base config, `task_type` is not initialised in it."
            )
        )
    TASK_CLASS = get_registered("task", config.task_type)
    task: TaskBase = TASK_CLASS(config)
    logger.info("Task initialised, ready to start")
    task.train()


if __name__ == "__main__":
    # 从命令行获取参数字符串
    import sys
    # cmd_args = " ".join(sys.argv[1:])  # 从第一个参数开始获取所有参数并拼接成字符串
    cmd_args = sys.argv[1:]
    # 调用 train 函数并将参数字符串传递给它
    train(cmd_args=cmd_args)


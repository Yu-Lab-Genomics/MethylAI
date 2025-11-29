import os
import functools
import importlib
import sys

def check_output_folder(folder):
    if not os.path.exists(folder):
        print(f'mkdir {folder}')
        os.makedirs(folder)

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def debug_methods(cls):
    """类装饰器，自动为所有方法添加调试输出"""
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, log_function_call(method))
    return cls

def load_config(config_file, dict_name):
    """从Python模块导入配置字典"""
    try:
        # 导入模块
        module = importlib.import_module(config_file.replace('.py', ''))
        # 获取字典
        config_dict = getattr(module, dict_name)
        return config_dict
    except ImportError as e:
        print(f"错误: 无法导入模块 {config_file}")
        print(f"请确保文件 {config_file} 在Python路径中")
        sys.exit(1)
    except AttributeError as e:
        print(f"错误: 模块中未找到字典 {dict_name}")
        print(f"可用的字典: {[name for name in dir(module) if not name.startswith('_')]}")
        sys.exit(1)












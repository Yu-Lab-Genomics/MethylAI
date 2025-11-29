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
    """通过文件路径导入配置字典"""
    try:
        # 获取模块名
        module_name = os.path.splitext(os.path.basename(config_file))[0]
        # 从文件路径加载模块
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 获取字典
        config_dict = getattr(module, dict_name)
        return config_dict
    except FileNotFoundError:
        print(f"错误: 找不到文件 {config_file}")
        sys.exit(1)
    except AttributeError as e:
        print(f"错误: 文件中未找到字典 {dict_name}")
        # 列出文件中所有不以_开头的属性
        available_dicts = [name for name in dir(module) if not name.startswith('_') and isinstance(getattr(module, name), dict)]
        print(f"可用的字典: {available_dicts}")
        sys.exit(1)













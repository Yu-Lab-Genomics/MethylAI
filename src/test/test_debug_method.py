import functools

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


@debug_methods
class MyClass:
    def method1(self):
        print("执行方法1")

    def method2(self, x):
        print(f"执行方法2，参数: {x}")
        return x * 2


# 测试
obj = MyClass()
obj.method1()
obj.method2(5)
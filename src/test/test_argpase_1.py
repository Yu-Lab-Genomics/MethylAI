import argparse


def main():
    parser = argparse.ArgumentParser(description='这是一个命令行参数解析示例')
    # 添加位置参数
    parser.add_argument('--file', nargs='*', type=int, default=[])
    # 解析参数
    args = parser.parse_args()
    # 使用参数
    print(f"输入文件: {args.file}")
    print(f"参数类型：{type(args.file)}")
    if args.file:
        print('yes')
    else:
        print('no')


if __name__ == "__main__":
    main()
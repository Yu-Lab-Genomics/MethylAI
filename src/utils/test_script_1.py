import argparse


def main():
    parser = argparse.ArgumentParser(description='这是一个命令行参数解析示例')

    # 添加位置参数
    parser.add_argument('input_file', help='输入文件路径')

    # 添加可选参数
    parser.add_argument('-o', '--output', help='输出文件路径', default='output.txt')
    parser.add_argument('-v', '--verbose', help='详细输出', action='store_true')
    parser.add_argument('--count', type=int, help='重复次数', default=1)
    parser.add_argument('--mode', choices=['fast', 'normal', 'slow'],
                        default='normal', help='运行模式')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output}")
    print(f"详细模式: {args.verbose}")
    print(f"重复次数: {args.count}")
    print(f"运行模式: {args.mode}")

    # 模拟处理
    for i in range(args.count):
        if args.verbose:
            print(f"处理第 {i + 1} 次...")


if __name__ == "__main__":
    main()
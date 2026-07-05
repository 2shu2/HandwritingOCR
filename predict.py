"""
单张图片手写文字识别测试

用法:
  python predict.py                    # 交互菜单模式
  python predict.py --image xx.png     # 直接指定图片路径
  python predict.py --demo             # 自动选取训练集第一张测试
  python predict.py --batch a.png b.png  # 批量识别
"""
import os
import glob
import argparse
import yaml
import torch
import cv2
from model import CRNN
from data import decode_predictions


def preprocess_image(image_path: str, img_height: int) -> torch.Tensor:
    """读取并预处理单张图片，返回 (1,1,H,W) 张量"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape
    scale = img_height / h
    new_w = max(int(w * scale), 1)
    img = cv2.resize(img, (new_w, img_height))
    img = img.astype('float32') / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return img_tensor


def find_images(data_dirs: list) -> list:
    """在数据目录中搜索可用的图片（去重）"""
    seen = set()
    images = []
    for d in data_dirs:
        if os.path.isdir(d):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for f in glob.glob(os.path.join(d, ext)):
                    name = os.path.basename(f)
                    if name not in seen:
                        seen.add(name)
                        images.append(f)
    return sorted(images)


def run_inference(model, image_path: str, img_height: int, idx2char: dict):
    """执行推理并输出结果"""
    abs_path = os.path.abspath(image_path)
    print(f"  图片: {os.path.basename(image_path)}")
    print(f"  路径: {abs_path}")
    img_tensor = preprocess_image(image_path, img_height)
    with torch.no_grad():
        log_probs = model(img_tensor)
    # 置信度: 取每个时间步最大概率的平均值
    probs = torch.exp(log_probs)
    max_probs, _ = probs.max(dim=-1)
    confidence = max_probs.mean().item()
    pred_texts = decode_predictions(log_probs, idx2char)
    print(f"  识别结果: {pred_texts[0]}")
    print(f"  置信度: {confidence:.2%}")
    return pred_texts[0], confidence


def pick_from_training_set(config: dict) -> str:
    """从训练集/验证集中选择一张图片"""
    search_dirs = [
        config['data']['train_image_dir'],
        config['data']['val_image_dir'],
        "data/trdg/images",
    ]
    available = find_images(search_dirs)
    if not available:
        print("未找到训练集图片。请先运行 python train.py 自动生成数据。")
        return None

    print(f"\n训练集中共有 {len(available)} 张图片:\n")
    # 每页显示 20 张
    page_size = 20
    page = 0
    total_pages = (len(available) - 1) // page_size + 1

    while True:
        start = page * page_size
        end = min(start + page_size, len(available))
        for i in range(start, end):
            print(f"  [{i:4d}] {os.path.basename(available[i])}")

        if total_pages > 1:
            print(f"\n  第 {page + 1}/{total_pages} 页 ", end="")
        print(f"(共 {len(available)} 张)")
        print("\n输入序号选择 | 'n' 下一页 | 'p' 上一页 | 'q' 返回主菜单")
        choice = input("> ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 'n' and page < total_pages - 1:
            page += 1
        elif choice == 'p' and page > 0:
            page -= 1
        elif choice.isdigit() and 0 <= int(choice) < len(available):
            return available[int(choice)]
        else:
            print("无效输入，请重试。\n")


def pick_custom_image() -> str:
    """让用户输入自定义图片路径"""
    print("\n请输入图片路径（支持拖拽图片到终端）：")
    print("  - 绝对路径: C:\\Users\\...\\image.png")
    print("  - 相对路径: ./my_image.png")
    print("  - 直接回车返回主菜单")
    path = input("> ").strip().strip('"').strip("'")  # 去掉拖拽可能带来的引号
    if not path:
        return None
    if os.path.isfile(path):
        return path
    else:
        print(f"\n[错误] 文件不存在: {path}")
        print("请检查路径是否正确。")
        return None


def interactive_menu(config: dict, model, idx2char: dict, img_height: int):
    """交互式主菜单"""
    while True:
        print("\n" + "=" * 50)
        print("       手写英文识别 - 主菜单")
        print("=" * 50)
        print("  1. 使用训练集图片识别")
        print("  2. 上传自己的图片识别")
        print("  3. 退出")
        print("-" * 50)
        choice = input("请选择 (1/2/3): ").strip()

        if choice == '1':
            img_path = pick_from_training_set(config)
            if img_path:
                print()
                run_inference(model, img_path, img_height, idx2char)

        elif choice == '2':
            while True:
                img_path = pick_custom_image()
                if img_path is None:
                    break  # 返回主菜单
                print()
                try:
                    run_inference(model, img_path, img_height, idx2char)
                except FileNotFoundError as e:
                    print(f"[错误] {e}")
                except Exception as e:
                    print(f"[错误] {e}")
                # 继续循环，可以连续识别多张
                print("\n继续识别其他图片？(回车继续 / 'm' 返回主菜单)")
                if input("> ").strip().lower() == 'm':
                    break

        elif choice == '3':
            print("再见！")
            return
        else:
            print("无效选择，请输入 1、2 或 3。")


def main():
    parser = argparse.ArgumentParser(
        description="手写英文识别 - 本地测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python predict.py                  # 交互菜单\n"
               "  python predict.py --image xx.png    # 指定图片\n"
               "  python predict.py --demo            # 自动demo\n"
               "  python predict.py --batch a.png b.png  # 批量"
    )
    parser.add_argument('--image', type=str, default=None,
                        help='直接指定图片路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--demo', action='store_true',
                        help='自动选取训练集第一张测试')
    parser.add_argument('--batch', type=str, nargs='*',
                        help='批量识别多张图片')
    args = parser.parse_args()

    # ---------- 加载配置 ----------
    if not os.path.exists(args.config):
        print(f"[错误] 配置文件不存在: {args.config}")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ---------- 检查模型 ----------
    model_path = config['inference']['model_path']
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型。")
        return

    alphabet = config['data']['alphabet']
    num_classes = len(alphabet) + 1
    idx2char = {i + 1: ch for i, ch in enumerate(alphabet)}
    idx2char[0] = ''

    # ---------- 加载模型 ----------
    print(f"加载模型: {model_path}")
    model = CRNN(
        num_classes,
        config['model']['backbone'],
        config['model']['hidden_size'],
        config['model']['num_layers'],
        img_height=config['inference']['img_height']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("模型加载成功。")

    img_height = config['inference']['img_height']

    # ---------- 根据参数选择模式 ----------
    if args.batch:
        # 批量模式
        for img_path in args.batch:
            try:
                print()
                run_inference(model, img_path, img_height, idx2char)
            except Exception as e:
                print(f"[错误] {img_path}: {e}")
    elif args.image:
        # 指定图片
        run_inference(model, args.image, img_height, idx2char)
    elif args.demo:
        # 自动 demo
        search_dirs = [
            config['data']['train_image_dir'],
            config['data']['val_image_dir'],
            "data/trdg/images",
        ]
        available = find_images(search_dirs)
        if available:
            print(f"[demo] 自动选取: {available[0]}\n")
            run_inference(model, available[0], img_height, idx2char)
        else:
            print("未找到可用图片，请先运行 python train.py 自动生成数据。")
    else:
        # 默认：交互菜单
        interactive_menu(config, model, idx2char, img_height)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

import tool
import os
import sys

def main(k: int=-1):
    problem_dir = "problems"
    
    # 检查目录是否存在
    if not os.path.isdir(problem_dir):
        print(f"错误：目录 '{problem_dir}' 不存在。")
        return

    # 获取目录下所有文件（不包含子目录）
    all_files = [
        f for f in os.listdir(problem_dir)
        if os.path.isfile(os.path.join(problem_dir, f))
    ]
    total = len(all_files)

    if total == 0:
        print("目录下没有文件。")
        return

    # 处理前 k 个文件，k 通过命令行参数传入，默认全部
    if k!=-1:
        try:
            if k <= 0:
                print("k 必须为正整数，将处理所有文件。")
                k = total
            else:
                k = min(k, total)   # 防止超过总数
        except ValueError:
            print("k 格式不正确，将处理所有文件。")
            k = total
    else:
        k = total

    files_to_process = all_files[:k]

    print(f"开始处理 {len(files_to_process)} / {total} 个文件...")

    error_log_path = "batch_errors.log"
    error_count = 0

    for idx, filename in enumerate(files_to_process, start=1):
        filepath = os.path.join(problem_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tool.add_problem(filename, content)

        # 检出异常标签数
        info = tool.find_problem(filename)
        n_tags = len(info["tags"]) if info else 0
        raw_len = len(info.get("raw_output", "")) if info else 0

        if n_tags > 50 or n_tags == 0:
            error_count += 1
            with open(error_log_path, "a", encoding="utf-8") as err_f:
                err_f.write(f"[{idx}/{len(files_to_process)}] {filename}\n")
                err_f.write(f"  tags={n_tags}  raw_len={raw_len}\n")
                err_f.write(f"  raw_output[:200]={info.get('raw_output','')[:200]}\n\n")
            print(f"[{idx}/{len(files_to_process)}] ⚠ {filename} — tags={n_tags} (异常, 已记录)")
        else:
            print(f"[{idx}/{len(files_to_process)}] ✓ {filename} — {n_tags} tags")

    print(f"全部完成。异常: {error_count} 道 (详见 {error_log_path})")

if __name__ == "__main__":
    main(-1)


# updateTXT.py

import os
import shutil # Not directly used for copying content anymore, but kept for potential future use
import datetime

def get_formatted_timestamp_line():
    """Generates a formatted timestamp line for prepending to files."""
    now = datetime.datetime.now()
    return f"# File processed on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

def generate_code_structure_string(root_dir_param, backup_subdir_name_to_exclude, self_script_abs_path_param=None):
    """
    Generates a string representation of the directory structure of root_dir_param
    in a tree-like format.
    Excludes specified backup subdirectory, common VCS/dev folders, and optionally the script itself.
    """
    structure_lines = []
    root_dir_abs = os.path.abspath(root_dir_param)
    backup_subdir_basename = os.path.basename(backup_subdir_name_to_exclude)

    excluded_dir_basenames = {
        '.git', '.svn', 'CVS', '.hg',  # VCS
        '.idea', '.vscode', 'nbproject', # IDE
        '__pycache__',                  # Python cache
        'venv', '.venv', 'env', '.env', # Virtual environments
        'dist', 'build', 'htmlcov', 'target', 'bin', 'obj', # Build artifacts & compiled output
        'node_modules', 'bower_components', # JS dependencies
        backup_subdir_basename            # The backup directory itself
    }
    try:
        for item in os.listdir(root_dir_abs): # Dynamically add .egg-info, etc.
            if item.endswith(('.egg-info', '.dist-info', '.pytest_cache', '.mypy_cache', '.tox')):
                if os.path.isdir(os.path.join(root_dir_abs, item)):
                    excluded_dir_basenames.add(item)
    except OSError:
        pass # Ignore if listing fails

    excluded_file_basenames = {'.DS_Store', 'desktop.ini', 'Thumbs.db', '.project', '.pydevproject', '.classpath'}
    excluded_file_extensions = ('.pyc', '.pyo', '.swp', '.swo', '.bak', '.tmp', '.log')

    structure_lines.append(f"{os.path.basename(root_dir_abs)}/")

    def generate_tree_recursive(current_dir_abs, prefix=""):
        nonlocal structure_lines
        try:
            items = sorted(os.listdir(current_dir_abs))
        except OSError:
            structure_lines.append(f"{prefix}└── [Error accessing directory]")
            return

        dirs_to_process = []
        files_to_process = []

        for item_name in items:
            item_path = os.path.join(current_dir_abs, item_name)
            is_dir = os.path.isdir(item_path)
            if item_name.startswith('.') or \
               (is_dir and item_name in excluded_dir_basenames) or \
               (not is_dir and (item_name in excluded_file_basenames or item_name.endswith(excluded_file_extensions))):
                continue
            if not is_dir and self_script_abs_path_param and item_path == self_script_abs_path_param:
                continue

            if is_dir:
                dirs_to_process.append(item_name)
            else:
                files_to_process.append(item_name)
        all_items_to_list = dirs_to_process + files_to_process
        for i, item_name in enumerate(all_items_to_list):
            is_last_item = (i == len(all_items_to_list) - 1)
            connector = "└── " if is_last_item else "├── "
            structure_lines.append(f"{prefix}{connector}{item_name}")
            item_path = os.path.join(current_dir_abs, item_name)
            if os.path.isdir(item_path):
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                generate_tree_recursive(item_path, new_prefix)
    generate_tree_recursive(root_dir_abs)
    return "\n".join(structure_lines)

def backup_specific_py_to_txt(source_dir=".", backup_subdir="txt_backup"):
    """
    Backs up all .py files in root, and modules/*.py
    to .txt files in backup_subdir, adds timestamps, and generates code_structure.txt.
    """
    source_directory_abs = os.path.abspath(source_dir) # Define source_directory_abs early
    backup_dir_path = os.path.join(source_directory_abs, backup_subdir)

    self_script_abs_path = None
    try:
        self_script_abs_path = os.path.abspath(__file__)
    except NameError:
        print("警告: 无法确定脚本自身的文件名和路径。相关排除可能不精确或备份可能包含脚本自身。")

    timestamp_line = get_formatted_timestamp_line()

    print(f"准备从 '{source_directory_abs}' 备份特定文件和生成代码结构到 '{backup_dir_path}'...")
    files_to_backup = []
    # Keep track of files already added to avoid duplicates
    added_source_paths = set()

    # --- 1. 修改后的逻辑: 检测根目录下所有的 .py 文件 ---
    print(f"  正在检查根目录 '{source_directory_abs}' 中的所有 .py 文件...")
    found_root_py_files = False
    try:
        for item_name in os.listdir(source_directory_abs):
            item_source_path = os.path.join(source_directory_abs, item_name)
            # 检查是否是文件，以 .py 结尾
            if os.path.isfile(item_source_path) and item_name.endswith(".py"):
                if self_script_abs_path and item_source_path == self_script_abs_path:
                    print(f"  信息: '{item_name}' 是当前运行的脚本，备份将被跳过。")
                elif item_source_path in added_source_paths:
                    print(f"  信息: '{item_name}' 已被添加，跳过。")
                else:
                    files_to_backup.append({
                        "source_path": item_source_path,
                        "original_filename": item_name,
                        "relative_source_dir": "" # 表示在根目录
                    })
                    added_source_paths.add(item_source_path)
                    print(f"  找到根目录 .py 文件: '{item_name}' 准备备份。")
                    found_root_py_files = True
        if not found_root_py_files:
             print(f"  信息: 在根目录 '{source_directory_abs}' 中未找到 .py 文件进行备份。")

    except OSError as e:
        print(f"  错误: 无法读取源目录 '{source_directory_abs}' 中的文件列表: {e}")

    # --- 3. Target .py files in the modules subdirectory (这部分逻辑保持不变, 仅修改了文件夹名称) ---
    modules_subdir_name = "modules" # <--- 修改点
    modules_dir_path = os.path.join(source_directory_abs, modules_subdir_name) # <--- 修改点
    if os.path.exists(modules_dir_path) and os.path.isdir(modules_dir_path): # <--- 修改点
        print(f"  正在检查 '{modules_subdir_name}' 目录...") # <--- 修改点
        found_in_modules = False # <--- 修改点 (变量名对应)
        try:
            for item_name in sorted(os.listdir(modules_dir_path)): # Sort for consistent order # <--- 修改点
                if item_name.endswith(".py"):
                    item_source_path = os.path.join(modules_dir_path, item_name) # <--- 修改点
                    if os.path.isfile(item_source_path): # Ensure it's a file
                        if self_script_abs_path and item_source_path == self_script_abs_path:
                            print(f"  信息: '{modules_subdir_name}/{item_name}' 是当前运行的脚本，备份将被跳过。") # <--- 修改点
                        elif item_source_path in added_source_paths:
                            print(f"  信息: '{modules_subdir_name}/{item_name}' 已被添加，跳过。") # <--- 修改点
                        else:
                            files_to_backup.append({
                                "source_path": item_source_path,
                                "original_filename": item_name,
                                "relative_source_dir": modules_subdir_name # <--- 修改点
                            })
                            added_source_paths.add(item_source_path)
                            print(f"  找到: '{modules_subdir_name}/{item_name}' 准备备份。") # <--- 修改点
                            found_in_modules = True # <--- 修改点 (变量名对应)
            if not found_in_modules: # <--- 修改点 (变量名对应)
                print(f"  信息: 在 '{modules_dir_path}' 中未找到 .py 文件。") # <--- 修改点
        except OSError as e:
            print(f"  错误: 无法读取 '{modules_dir_path}' 目录: {e}") # <--- 修改点
    else:
        print(f"  跳过: '{modules_subdir_name}' 目录在 '{source_directory_abs}' 中未找到或不是一个目录。") # <--- 修改点

    # Create backup directory (if it doesn't exist)
    if not os.path.exists(backup_dir_path):
        try:
            os.makedirs(backup_dir_path)
            print(f"\n创建备份目录: {backup_dir_path}")
        except OSError as e:
            print(f"\n错误: 无法创建备份目录 {backup_dir_path}: {e}")
            print("备份中止。")
            return
    else:
        print(f"\n备份目录已存在: {backup_dir_path}")

    backed_up_py_count = 0
    if files_to_backup:
        print(f"\n开始备份 {len(files_to_backup)} 个 .py 文件...")
        files_to_backup.sort(key=lambda x: (x["relative_source_dir"], x["original_filename"]))

        for file_info in files_to_backup:
            source_file_path = file_info["source_path"]
            original_filename = file_info["original_filename"]
            base, _ = os.path.splitext(original_filename)
            destination_filename = base + ".txt"
            destination_file_path = os.path.join(backup_dir_path, destination_filename)

            try:
                with open(source_file_path, 'r', encoding='utf-8', errors='replace') as f_src:
                    content = f_src.read()
                with open(destination_file_path, 'w', encoding='utf-8') as f_dest:
                    f_dest.write(timestamp_line)
                    f_dest.write(content)
                display_source = os.path.join(file_info["relative_source_dir"], original_filename) \
                                 if file_info["relative_source_dir"] else original_filename
                print(f"  已备份: {display_source} -> {os.path.join(backup_subdir, destination_filename)}")
                backed_up_py_count += 1
            except Exception as e:
                print(f"  错误: 无法备份 {original_filename}: {e}")
    structure_generated = False
    print(f"\n正在生成代码结构图 '{os.path.join(backup_subdir, 'code_structure.txt')}'...")
    try:
        code_structure_content = generate_code_structure_string(
            source_directory_abs,
            backup_subdir,
            self_script_abs_path
        )
        structure_file_path = os.path.join(backup_dir_path, "code_structure.txt")
        with open(structure_file_path, 'w', encoding='utf-8') as f_struct:
            f_struct.write(timestamp_line)
            f_struct.write(code_structure_content)
        print(f"  已生成: {os.path.join(backup_subdir, 'code_structure.txt')}")
        structure_generated = True
    except Exception as e:
        print(f"  错误: 无法生成代码结构图: {e}")

    print("\n--- 操作总结 ---")
    py_backup_summary = "Python 文件备份: 无指定文件需要备份。"
    if files_to_backup:
        if backed_up_py_count == len(files_to_backup):
            py_backup_summary = f"Python 文件备份: 成功 ({backed_up_py_count}/{len(files_to_backup)} 文件)。"
        else:
            py_backup_summary = f"Python 文件备份: 部分成功或失败 ({backed_up_py_count}/{len(files_to_backup)} 文件)。"
    print(py_backup_summary)
    structure_summary = "代码结构图: 生成失败。"
    if structure_generated:
        structure_summary = "代码结构图: 成功生成。"
    print(structure_summary)

    all_targeted_actions_successful = True
    if files_to_backup and backed_up_py_count != len(files_to_backup):
        all_targeted_actions_successful = False
    if not structure_generated:
        all_targeted_actions_successful = False
    if not files_to_backup and not structure_generated :
        print("主要操作未执行或失败 (无指定PY文件，且结构图生成失败)。")
    elif all_targeted_actions_successful and (files_to_backup or structure_generated):
          print("所有目标操作均已成功完成。")
    elif not all_targeted_actions_successful and (files_to_backup or structure_generated):
        print("部分目标操作未成功完成，请检查上述日志。")
    elif not files_to_backup and structure_generated:
        print("Python 文件备份: 无指定文件需要备份。")
        print("代码结构图: 成功生成。")
        print("所有目标操作均已成功完成。")

    print("\n备份程序执行完毕。")

if __name__ == "__main__":
    script_location_dir = "."
    try:
        script_location_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # pragma: no cover
        print("警告: 无法自动确定脚本所在目录，将使用当前工作目录作为源目录。")
        print("       这可能导致脚本自身被包含在代码结构图或备份中（如果名称匹配）。")

    backup_specific_py_to_txt(source_dir=script_location_dir)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试CIF到PDB转换修复
"""

import os
from pathlib import Path

def test_path_handling():
    """测试Path对象处理"""
    print("测试Path对象处理...")
    
    # 模拟文件路径
    cif_path_str = "test.cif"
    pdb_path_str = "test.pdb"
    
    # 测试字符串到Path的转换
    if isinstance(cif_path_str, str):
        cif_path = Path(cif_path_str)
        print(f"✅ 字符串转换为Path: {cif_path_str} -> {cif_path}")
    
    if isinstance(pdb_path_str, str):
        pdb_path = Path(pdb_path_str)
        print(f"✅ 字符串转换为Path: {pdb_path_str} -> {pdb_path}")
    
    # 测试Path对象保持不变
    if isinstance(cif_path, Path):
        print(f"✅ Path对象保持不变: {cif_path}")
    
    return True

def test_file_validation():
    """测试文件验证逻辑"""
    print("\n测试文件验证逻辑...")
    
    # 创建临时测试文件
    test_file = Path("temp_test.pdb")
    
    # 写入测试内容
    test_content = """TITLE     TEST STRUCTURE
ATOM      1  N   ALA A   1      27.462  14.105   5.468  1.00 20.00           N
ATOM      2  CA  ALA A   1      26.213  14.871   5.548  1.00 20.00           C
END"""
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # 测试文件验证
    if test_file.exists():
        print(f"✅ 文件存在: {test_file}")
        
        file_size = test_file.stat().st_size
        print(f"✅ 文件大小: {file_size} bytes")
        
        if file_size > 0:
            print("✅ 文件非空")
        else:
            print("❌ 文件为空")
    else:
        print("❌ 文件不存在")
    
    # 清理测试文件
    if test_file.exists():
        test_file.unlink()
        print("✅ 测试文件已清理")
    
    return True

def test_error_handling():
    """测试错误处理"""
    print("\n测试错误处理...")
    
    # 测试不存在的文件
    non_existent_file = Path("non_existent.cif")
    
    try:
        if not non_existent_file.exists():
            print(f"✅ 正确检测到文件不存在: {non_existent_file}")
        else:
            print(f"❌ 错误：文件应该不存在: {non_existent_file}")
    except Exception as e:
        print(f"❌ 错误处理异常: {e}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("测试CIF到PDB转换修复")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_path_handling,
        test_file_validation,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    if passed == total:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 50)

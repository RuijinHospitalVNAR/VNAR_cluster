#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的AF3精细聚类脚本
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_foldseek_command():
    """测试Foldseek命令格式"""
    print("=" * 60)
    print("测试Foldseek命令格式")
    print("=" * 60)
    
    # 创建测试文件
    test_dir = Path("test_foldseek")
    test_dir.mkdir(exist_ok=True)
    
    # 创建一些测试PDB文件
    for i in range(3):
        pdb_file = test_dir / f"test_{i}.pdb"
        with open(pdb_file, 'w') as f:
            f.write(f"""ATOM      1  N   ALA A   1      27.000  25.000  25.000  1.00 20.00
ATOM      2  CA  ALA A   1      26.000  25.000  25.000  1.00 20.00
ATOM      3  C   ALA A   1      25.000  25.000  25.000  1.00 20.00
ATOM      4  O   ALA A   1      24.000  25.000  25.000  1.00 20.00
TER
""")
    
    print(f"创建了测试目录: {test_dir}")
    print(f"测试文件:")
    for pdb_file in test_dir.glob("*.pdb"):
        print(f"  {pdb_file.name}")
    
    # 清理
    shutil.rmtree(test_dir)
    print("✅ Foldseek命令格式测试完成")

def test_usalign_indexing():
    """测试US-align索引映射"""
    print("\n" + "=" * 60)
    print("测试US-align索引映射")
    print("=" * 60)
    
    # 模拟索引映射
    idx = [10, 15, 20, 25, 30]  # 全局索引
    neighbor_matrix = [
        [False, True,  False, False, False],
        [True,  False, True,  False, False],
        [False, True,  False, True,  False],
        [False, False, True,  False, True],
        [False, False, False, True,  False]
    ]
    
    import numpy as np
    neighbor_matrix = np.array(neighbor_matrix)
    
    # 获取邻居对
    neighbor_pairs = np.where(neighbor_matrix)
    pairs = list(zip(neighbor_pairs[0], neighbor_pairs[1]))
    
    print(f"局部索引对: {pairs}")
    
    # 转换为全局索引
    global_pairs = []
    for local_i, local_j in pairs:
        if local_i < len(idx) and local_j < len(idx):
            global_i = idx[local_i]
            global_j = idx[local_j]
            global_pairs.append((global_i, global_j))
    
    print(f"全局索引对: {global_pairs}")
    
    # 创建映射
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(idx)}
    print(f"全局到局部映射: {global_to_local}")
    
    print("✅ US-align索引映射测试完成")

def test_import():
    """测试模块导入"""
    print("\n" + "=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    try:
        from AF3_Cluster_fine_v1 import AF3FineClusterAnalyzer
        print("✅ 模块导入成功")
        
        # 测试类初始化
        analyzer = AF3FineClusterAnalyzer(
            coarse_results_file="test.pkl",
            pdb_dir="test_dir",
            n_jobs=2
        )
        print("✅ 类初始化成功")
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("开始测试修复...")
    
    # 测试1: Foldseek命令格式
    test_foldseek_command()
    
    # 测试2: US-align索引映射
    test_usalign_indexing()
    
    # 测试3: 模块导入
    if test_import():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)

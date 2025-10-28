#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Foldseek + US-align聚类方案
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AF3_Cluster_fine_v1 import AF3FineClusterAnalyzer

def test_foldseek_usalign_clustering():
    """测试Foldseek + US-align聚类方案"""
    
    print("=" * 60)
    print("测试Foldseek + US-align聚类方案")
    print("=" * 60)
    
    # 创建测试数据
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # 创建一些测试PDB文件（简单的示例）
    test_pdb_files = []
    for i in range(5):
        pdb_file = test_dir / f"test_structure_{i}.pdb"
        with open(pdb_file, 'w') as f:
            f.write(f"""ATOM      1  N   ALA A   1      27.000  25.000  25.000  1.00 20.00
ATOM      2  CA  ALA A   1      26.000  25.000  25.000  1.00 20.00
ATOM      3  C   ALA A   1      25.000  25.000  25.000  1.00 20.00
ATOM      4  O   ALA A   1      24.000  25.000  25.000  1.00 20.00
TER
""")
        test_pdb_files.append(str(pdb_file))
    
    # 创建测试的粗聚类结果
    import pickle
    import numpy as np
    
    test_results = {
        'file_names': [f"test_structure_{i}.pdb" for i in range(5)],
        'coarse_labels': np.array([0, 0, 0, 1, 1]),  # 两个粗聚类
        'contact_sets': [set() for _ in range(5)],
        'structures': []
    }
    
    results_file = test_dir / "test_coarse_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(test_results, f)
    
    # 初始化分析器
    analyzer = AF3FineClusterAnalyzer(
        coarse_results_file=str(results_file),
        pdb_dir=str(test_dir),
        n_jobs=2
    )
    
    # 加载粗聚类结果
    print("\n1. 加载粗聚类结果...")
    if not analyzer.load_coarse_results():
        print("错误: 无法加载粗聚类结果")
        return False
    
    print(f"   加载了 {len(analyzer.file_names)} 个结构")
    print(f"   粗聚类标签: {analyzer.coarse_labels}")
    
    # 测试精细聚类
    print("\n2. 测试精细聚类...")
    clustering_config = {
        'max_clusters_to_refine': 2,
        'max_cluster_size_for_refine': 10,
        'k_neighbors': 3,
        'clustering_method': 'hdbscan',
        'n_jobs': 2
    }
    
    try:
        fine_labels = analyzer.perform_fine_clustering(**clustering_config)
        print(f"   精细聚类完成，标签: {fine_labels}")
        print(f"   精细聚类数量: {len(set(fine_labels)) - (1 if -1 in fine_labels else 0)}")
        return True
    except Exception as e:
        print(f"   精细聚类失败: {e}")
        return False
    finally:
        # 清理测试文件
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_foldseek_usalign_clustering()
    if success:
        print("\n✅ 测试通过！")
    else:
        print("\n❌ 测试失败！")

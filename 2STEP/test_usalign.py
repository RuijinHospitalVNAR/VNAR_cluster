#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
US-align测试脚本
用于验证US-align是否能正常工作
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# US-align路径
USALIGN_CMD = os.environ.get('USALIGN_CMD', '/mnt/share/public/USalign')

def test_usalign_basic():
    """测试US-align基本功能"""
    print("=" * 50)
    print("测试US-align基本功能")
    print("=" * 50)
    
    # 检查US-align是否存在
    if not os.path.exists(USALIGN_CMD):
        print(f"❌ US-align不存在: {USALIGN_CMD}")
        return False
    
    print(f"✅ 找到US-align: {USALIGN_CMD}")
    
    # 测试US-align版本
    try:
        result = subprocess.run([USALIGN_CMD, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ US-align可以正常运行")
            print(f"帮助信息前几行:\n{result.stdout[:200]}...")
        else:
            print(f"❌ US-align运行失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ US-align测试失败: {e}")
        return False
    
    return True

def test_usalign_with_files():
    """测试US-align与文件"""
    print("\n" + "=" * 50)
    print("测试US-align与文件")
    print("=" * 50)
    
    # 查找测试文件
    test_files = []
    
    # 优先查找粗聚类文件夹中的文件
    coarse_dir = "/mnt/share/chufan/VNAR_SH3/2d4_PI3Ka_cluster/2d4_pi3ka_renamed_af3_coarse_cluster_20250826_123519/coarse_clusters/"
    if os.path.exists(coarse_dir):
        for cluster_dir in os.listdir(coarse_dir):
            cluster_path = os.path.join(coarse_dir, cluster_dir)
            if os.path.isdir(cluster_path):
                for file in os.listdir(cluster_path):
                    if file.endswith(('.pdb', '.cif')):
                        test_files.append(os.path.join(cluster_path, file))
                        if len(test_files) >= 2:
                            break
                if len(test_files) >= 2:
                    break
    
    # 如果粗聚类文件夹中没有找到，查找原始PDB文件
    if len(test_files) < 2:
        pdb_dir = "./2d4_pi3ka_renamed"
        if os.path.exists(pdb_dir):
            for file in os.listdir(pdb_dir):
                if file.endswith(('.pdb', '.cif')):
                    test_files.append(os.path.join(pdb_dir, file))
                    if len(test_files) >= 2:
                        break
    
    if len(test_files) < 2:
        print("❌ 找不到足够的测试文件")
        return False
    
    print(f"✅ 找到测试文件: {test_files[0]}, {test_files[1]}")
    
    # 测试US-align
    try:
        result = subprocess.run([
            USALIGN_CMD, test_files[0], test_files[1]
        ], capture_output=True, text=True, timeout=30)
        
        print(f"返回码: {result.returncode}")
        print(f"标准输出:\n{result.stdout}")
        print(f"标准错误:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✅ US-align成功运行")
            
            # 尝试解析RMSD
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'RMSD=' in line:
                    print(f"✅ 找到RMSD行: {line}")
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'RMSD=':
                            if i + 1 < len(parts):
                                try:
                                    rmsd_value = float(parts[i + 1])
                                    print(f"✅ 成功解析RMSD: {rmsd_value}")
                                    return True
                                except ValueError:
                                    continue
                    # 尝试其他格式
                    for part in parts:
                        if part.startswith('RMSD='):
                            try:
                                rmsd_value = float(part.split('=')[1])
                                print(f"✅ 成功解析RMSD: {rmsd_value}")
                                return True
                            except (ValueError, IndexError):
                                continue
            
            print("❌ 无法解析RMSD值")
            return False
        else:
            print("❌ US-align运行失败")
            return False
            
    except Exception as e:
        print(f"❌ US-align测试失败: {e}")
        return False

def main():
    """主函数"""
    print("US-align测试脚本")
    print("=" * 50)
    
    # 基本测试
    if not test_usalign_basic():
        print("\n❌ 基本测试失败")
        return
    
    # 文件测试
    if not test_usalign_with_files():
        print("\n❌ 文件测试失败")
        return
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    main()

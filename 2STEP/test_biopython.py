#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试biopython导入
"""

def test_biopython_import():
    """测试biopython导入"""
    try:
        import Bio
        print("✅ Bio模块导入成功")
        print(f"Bio版本: {Bio.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Bio模块导入失败: {e}")
        return False

def test_biopython_components():
    """测试biopython组件"""
    try:
        from Bio import PDB
        print("✅ Bio.PDB导入成功")
        
        from Bio.PDB import PDBParser, MMCIFParser
        print("✅ Bio.PDB组件导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ Bio.PDB组件导入失败: {e}")
        return False

if __name__ == "__main__":
    print("测试biopython导入...")
    
    if test_biopython_import() and test_biopython_components():
        print("✅ 所有biopython测试通过")
    else:
        print("❌ biopython测试失败")

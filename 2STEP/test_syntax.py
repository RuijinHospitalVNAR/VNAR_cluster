#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试AF3_Cluster_fine_v1.py的语法
"""

import sys
import os

def test_syntax():
    """测试Python文件的语法"""
    try:
        # 尝试导入模块
        from AF3_Cluster_fine_v1 import AF3FineClusterAnalyzer
        print("✅ 语法检查通过！")
        print("✅ 模块导入成功！")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)

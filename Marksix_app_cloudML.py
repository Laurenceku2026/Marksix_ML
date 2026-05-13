# ============================================================
# 六合彩AI智能选号工具 - 云端ML完整版 v7.1
# 第1部分：导入、配置、常量、Supabase连接、基础工具函数
# 
# 修复内容：
#   1. 数据加载时按期次数字排序（解决字符串排序问题）
#   2. 日期转换为Excel序列数（解决数据库类型不匹配）
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import json
import os
import hashlib
import hmac
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client

# 尝试导入机器学习库
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="六合彩AI分析工具 - 云端ML完整版 v7.1",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .stDataFrame { text-align: center; }
    .stDataFrame table { text-align: center; width: 100%; }
    .stDataFrame th { text-align: center !important; }
    .stDataFrame td { text-align: center !important; }
    .stMetric { text-align: center; }
    .stNumberInput input { text-align: center; }
    .stCheckbox { margin-top: 10px; }
    .stAlert { font-size: 0.9rem; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 常量定义 ====================
RED_NUMBERS = list(range(1, 50))  # 1-49

# 7分区定义（六合彩专用）
ZONES = {
    1: {'name': 'A区', 'range': '01-07', 'numbers': list(range(1, 8))},
    2: {'name': 'B区', 'range': '08-14', 'numbers': list(range(8, 15))},
    3: {'name': 'C区', 'range': '15-21', 'numbers': list(range(15, 22))},
    4: {'name': 'D区', 'range': '22-28', 'numbers': list(range(22, 29))},
    5: {'name': 'E区', 'range': '29-35', 'numbers': list(range(29, 36))},
    6: {'name': 'F区', 'range': '36-42', 'numbers': list(range(36, 43))},
    7: {'name': 'G区', 'range': '43-49', 'numbers': list(range(43, 50))}
}

# 理论均值（7码）：(1+49)/2 × 7 = 175
EXPECTED_SUM = 175
SUM_STD = 35  # 标准差约35

# 默认训练期数配置
DEFAULT_TRAIN_WINDOWS = {
    "方法1": 50,
    "方法2": 50,
    "方法3": 100,
    "方法4": 200,
    "方法5": 200
}

# 训练期数范围
TRAIN_WINDOW_RANGES = {
    "方法1": (30, 200),
    "方法2": (30, 200),
    "方法3": (50, 300),
    "方法4": (100, 500),
    "方法5": (100, 500)
}

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")

# DeepSeek API 限流
_last_deepseek_call = 0
DEEPSEEK_RATE_LIMIT = 3

# ==================== Supabase 初始化 ====================
def init_supabase() -> Optional[Client]:
    """初始化Supabase连接"""
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not supabase_url or not supabase_key:
            return None
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Supabase初始化失败: {e}")
        return None


def datetime_to_excel_serial(dt: datetime) -> int:
    """将datetime转换为Excel序列数（整数）"""
    base_date = datetime(1900, 1, 1)
    delta = dt - base_date
    days = delta.days + 2  # Excel的1900年日期系统有偏移
    return days


def date_string_to_excel_serial(date_str: str) -> Optional[int]:
    """将日期字符串转换为Excel序列数"""
    if not date_str:
        return None
    try:
        # 提取日期部分（去除时间）
        if ' ' in date_str:
            date_str = date_str[:10]
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return datetime_to_excel_serial(dt)
    except:
        return None


def excel_serial_to_date_string(serial: int) -> str:
    """将Excel序列数转换为日期字符串"""
    try:
        base_date = datetime(1900, 1, 1)
        dt = base_date + timedelta(days=serial - 2)
        return dt.strftime('%Y-%m-%d')
    except:
        return ""


# ==================== 分区函数 ====================
def get_zone(num: int) -> int:
    """获取号码所在分区（1-7）"""
    return (num - 1) // 7 + 1


def get_zone_numbers(zone: int) -> List[int]:
    """获取指定分区的所有号码"""
    start = (zone - 1) * 7 + 1
    end = start + 6
    return list(range(start, end + 1))


def calculate_zone_heat(draws: List[Dict], last_n: int = 20) -> Tuple[Dict[int, float], Dict[int, int]]:
    """计算分区热度"""
    zone_hits = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    zone_trend = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    
    recent_draws = draws[-last_n:] if len(draws) >= last_n else draws
    
    for idx, draw in enumerate(recent_draws):
        for num in draw['numbers']:
            zone = get_zone(num)
            zone_hits[zone] += 1
            zone_trend[zone].append(idx)
    
    zone_scores = {}
    for zone in range(1, 8):
        hits = zone_hits[zone]
        recent_weight = 0
        for pos in zone_trend[zone][-5:]:
            recent_weight += (5 - (last_n - pos)) if (last_n - pos) < 5 else 0
        zone_scores[zone] = hits * 1.0 + recent_weight * 0.5
    
    return zone_scores, zone_hits


def get_hot_zones(zone_scores: Dict[int, float], num_hot_zones: int = 3) -> List[int]:
    """获取热门分区"""
    sorted_zones = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
    return [zone for zone, score in sorted_zones[:num_hot_zones]]


# ==================== 规律加权函数（基于269期数据验证） ====================
def calculate_consecutive_length(reds: List[int], target_num: int) -> int:
    """
    计算如果加入target_num，连号组的最大长度
    返回0表示不形成连号
    """
    if target_num in reds:
        return 0
    
    test_set = set(reds) | {target_num}
    sorted_test = sorted(test_set)
    
    max_len = 1
    current_len = 1
    
    for i in range(1, len(sorted_test)):
        if sorted_test[i] == sorted_test[i-1] + 1:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1
    
    return max_len if max_len > 1 else 0


def calculate_six_mark_boost(num: int, last_reds: List[int], last_special: Optional[int]) -> float:
    """
    计算六合彩规律加权系数
    基于269期历史数据验证
    
    参数:
        num: 要计算的号码
        last_reds: 上期正码列表（6个）
        last_special: 上期特码
    """
    if not last_reds:
        return 1.0
    
    boost = 1.0
    all_last_numbers = last_reds + ([last_special] if last_special else [])
    last_reds_sorted = sorted(last_reds)
    
    # ========== 1. 边号加权 ==========
    is_edge = False
    for r in all_last_numbers:
        if abs(num - r) == 1:
            is_edge = True
            break
    
    if is_edge:
        # 区分正码边号和特码边号
        is_edge_to_special = last_special and abs(num - last_special) == 1
        if is_edge_to_special:
            boost *= 1.2  # 特码边号
        else:
            boost *= 1.3  # 正码边号
    
    # ========== 2. 夹号加权（重点） ==========
    for i in range(len(last_reds_sorted) - 1):
        left = last_reds_sorted[i]
        right = last_reds_sorted[i + 1]
        
        if left < num < right:
            gap = right - left
            
            if gap == 2:
                boost *= 2.5   # 间隔2，最强
            elif gap == 3:
                boost *= 1.8   # 间隔3，次强
            elif gap == 4:
                boost *= 1.3   # 间隔4
            else:
                boost *= 1.1   # 间隔5+
            break
    
    # ========== 3. 连号加权 ==========
    test_reds = sorted(set(last_reds_sorted) | {num})
    consecutive_len = calculate_consecutive_length(test_reds, num)
    
    if consecutive_len >= 3:
        boost *= 1.5   # 3连以上
    elif consecutive_len == 2:
        boost *= 1.2   # 2连
    
    # ========== 4. 重号加权（仅特码） ==========
    if last_special and num == last_special:
        boost *= 1.15   # 特码重复，轻度加权
    
    # 正码重复不加权（×1.0）
    
    # 最高限制3倍
    return min(boost, 3.0)


# ==================== Supabase 数据操作（修复版） ====================
def save_draws_to_supabase(draws: List[Dict]) -> bool:
    """保存开奖数据到Supabase（覆盖保存）- 修复日期格式"""
    supabase = init_supabase()
    if supabase is None:
        return False
    try:
        # 清空现有数据
        supabase.schema('marksix_schema').table('marksix_draws').delete().neq("id", 0).execute()
        
        # 插入新数据
        for draw in draws:
            # 日期处理：确保是 YYYY-MM-DD 格式
            date_value = draw.get('date')
            date_str = None
            if date_value:
                if isinstance(date_value, str):
                    # 已经是字符串，提取日期部分
                    date_str = date_value.split()[0] if ' ' in date_value else date_value[:10]
                elif isinstance(date_value, datetime):
                    date_str = date_value.strftime('%Y-%m-%d')
                else:
                    # 如果是数字，尝试转换（但不应发生）
                    date_str = str(date_value)
            
            # 期次处理
            period = draw.get('period')
            if isinstance(period, str) and period.isdigit():
                period = int(period)
            
            data = {
                "period": period if period else 0,
                "date": date_str,  # 使用字符串格式 YYYY-MM-DD
                "numbers": draw['numbers'],
                "special": draw.get('special', 0),
                "sum_value": draw['sum']
            }
            
            print(f"保存数据: period={data['period']}, date={data['date']}, numbers={data['numbers']}, special={data['special']}")
            
            supabase.schema('marksix_schema').table('marksix_draws').insert(data).execute()
        
        return True
    except Exception as e:
        st.error(f"保存到Supabase失败: {e}")
        return False

def load_draws_from_supabase() -> Optional[List[Dict]]:
    """从Supabase加载开奖数据（按期次数字排序）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    try:
        response = supabase.schema('marksix_schema').table('marksix_draws')\
            .select("*")\
            .execute()
        
        if not response.data:
            return None
        
        draws = []
        for row in response.data:
            # 日期处理：数据库返回的可能是字符串或日期对象
            date_value = row.get('date')
            date_str = None
            if date_value:
                if isinstance(date_value, str):
                    date_str = date_value[:10] if ' ' in date_value else date_value
                elif isinstance(date_value, datetime):
                    date_str = date_value.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_value)[:10]
            
            draws.append({
                'period': row.get('period'),
                'date': date_str,
                'numbers': row['numbers'],
                'special': row.get('special'),
                'sum': row['sum_value']
            })
        
        # 按期次数字排序
        draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
        
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None

def load_recent_draws_from_supabase(limit: int = 300) -> Optional[List[Dict]]:
    """加载最近N期数据（按期次降序取N条，再反转）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    try:
        # 按期次降序取N条
        response = supabase.schema('marksix_schema').table('marksix_draws')\
            .select("*")\
            .order("period", desc=True)\
            .limit(limit)\
            .execute()
        
        if not response.data:
            return None
        
        draws = []
        for row in reversed(response.data):  # 反转回升序
            date_value = row.get('date')
            date_str = None
            if date_value and isinstance(date_value, int):
                date_str = excel_serial_to_date_string(date_value)
            elif date_value and isinstance(date_value, str):
                date_str = date_value[:10] if ' ' in date_value else date_value
            
            draws.append({
                'period': row.get('period'),
                'date': date_str,
                'numbers': row['numbers'],
                'special': row.get('special'),
                'sum': row['sum_value']
            })
        
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None


# ==================== 辅助函数 ====================
def convert_6sum_to_7sum(sum_6: int) -> int:
    """将6码和值转换为7码和值"""
    return int(sum_6 * 7 / 6)


def get_target_sum_by_numbers_count(num_count: int) -> int:
    """根据号码个数获取理论均值"""
    if num_count == 7:
        return 175
    elif num_count == 6:
        return 150
    else:
        return int((1 + 49) / 2 * num_count)


def has_consecutive_or_jump(nums: List[int]) -> bool:
    """检查是否有连号或跳号（已废弃，保留用于兼容）"""
    nums = sorted(nums)
    for i in range(len(nums)-1):
        diff = nums[i+1] - nums[i]
        if diff == 1 or diff == 2:
            return True
    return False


def get_dynamic_sum_range(draws: List[Dict], num_count: int, window: int = 4, sigma_factor: float = 0.5) -> Tuple[int, int, str, str, float, float, float]:
    """
    动态和值预测
    
    返回: (目标值, 容差, 方向, 方向描述, 长期均值, 长期标准差, 短期均值)
    """
    recent_draws = draws[-100:] if len(draws) >= 100 else draws
    all_sum_7 = [convert_6sum_to_7sum(d['sum']) for d in recent_draws]
    long_term_mean = np.mean(all_sum_7) if all_sum_7 else 175
    long_term_std = np.std(all_sum_7) if len(all_sum_7) > 1 else 35
    
    short_draws = draws[-window:] if len(draws) >= window else draws
    short_sum_7 = [convert_6sum_to_7sum(d['sum']) for d in short_draws]
    short_mean = np.mean(short_sum_7) if short_sum_7 else long_term_mean
    
    threshold = long_term_std * 0.1
    
    if short_mean > long_term_mean + threshold:
        target = long_term_mean - long_term_std * sigma_factor
        direction = "偏大回归"
        direction_desc = f"📈 偏大 (最近{window}期均值={short_mean:.1f} > 长期均值+{threshold:.1f})"
    elif short_mean < long_term_mean - threshold:
        target = long_term_mean + long_term_std * sigma_factor
        direction = "偏小回归"
        direction_desc = f"📉 偏小 (最近{window}期均值={short_mean:.1f} < 长期均值-{threshold:.1f})"
    else:
        target = long_term_mean
        direction = "正常"
        direction_desc = f"⚖️ 正常 (最近{window}期均值={short_mean:.1f} 在 ±{threshold:.1f} 范围内)"
    
    tolerance = max(10, int(long_term_std * sigma_factor))
    
    if num_count != 7:
        target = int(target * num_count / 7)
        tolerance = int(tolerance * num_count / 7)
    
    return int(target), tolerance, direction, direction_desc, long_term_mean, long_term_std, short_mean


def parse_datetime_string(datetime_str: str) -> Optional[int]:
    """解析日期时间字符串为Excel序列数"""
    datetime_str = datetime_str.strip()
    if not datetime_str:
        return None
    
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
        "%Y%m%d %H:%M:%S", "%Y%m%d %H:%M", "%Y%m%d",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(datetime_str, fmt)
            return datetime_to_excel_serial(dt)
        except ValueError:
            continue
    
    return None


def get_next_period(draws: List[Dict]) -> str:
    """获取下一期期号"""
    if not draws:
        return "未知"
    latest_period = draws[-1].get('period', '')
    if latest_period and str(latest_period).isdigit():
        return str(int(latest_period) + 1)
    return "未知"


def get_sorted_draws(draws: List[Dict]) -> List[Dict]:
    """按期次数字排序（升序）"""
    return sorted(draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)


def get_latest_and_oldest(draws: List[Dict]) -> Tuple[Dict, Dict]:
    """获取最新和最旧期次"""
    sorted_draws = get_sorted_draws(draws)
    return sorted_draws[-1], sorted_draws[0] if sorted_draws else ({}, {})


print("第1部分加载完成 (v7.1 - 修复排序和日期问题)")
print("=" * 60)
print("请确认第1部分代码，输入 CONFIRM 后继续第2部分")
print("=" * 60)
# ============================================================
# ============================================================
# 第2部分：管理员页面 + 登录验证 + 数据编辑器
# 版本：v7.1
# 
# 修复内容：
#   1. 数据显示按期次数字降序（最新在上）
#   2. 保存时正确转换日期格式
#   3. 修复Excel上传后的排序问题
# ============================================================

# ==================== 管理员密码验证 ====================
def check_password(password: str) -> bool:
    """验证管理员密码"""
    return hmac.compare_digest(password, "Ku_product$2026")


def admin_login():
    """管理员登录表单"""
    with st.form("admin_login_form"):
        username = st.text_input("用户名", key="admin_username")
        password = st.text_input("密码", type="password", key="admin_password")
        submitted = st.form_submit_button("登录")
        
        if submitted:
            if username == "Laurence_ku" and check_password(password):
                st.session_state['admin_logged_in'] = True
                st.session_state['show_admin'] = False
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")


def admin_logout():
    """管理员退出登录"""
    if st.button("退出登录", key="logout_btn"):
        st.session_state['admin_logged_in'] = False
        st.session_state['show_admin'] = False
        st.rerun()


# ==================== 数据解析函数 ====================
def parse_pasted_data(text: str) -> List[Dict]:
    """解析粘贴的数据文本"""
    lines = text.strip().split('\n')
    draws = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.replace(',', '\t').replace(' ', '\t').split('\t')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 9:
            try:
                nums = []
                for i in range(2, 9):
                    if i < len(parts):
                        num = int(float(parts[i]))
                        nums.append(num)
                if len(nums) == 7:
                    draws.append({
                        'period': int(parts[0]) if parts[0].isdigit() else parts[0],
                        'date': parts[1] if len(parts) > 1 else None,
                        'numbers': sorted(nums[:6]),
                        'special': nums[6],
                        'sum': sum(nums[:6])
                    })
            except (ValueError, IndexError):
                continue
    
    # 按期次数字排序
    draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
    return draws


def parse_multi_draws_for_checking(text: str, max_draws: int = 5) -> List[Dict]:
    """解析多期开奖数据用于查奖"""
    lines = text.strip().split('\n')
    draws = []
    for line in lines:
        if len(draws) >= max_draws:
            break
        if not line.strip():
            continue
        parts = line.replace(',', '\t').replace(' ', '\t').split('\t')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 9:
            try:
                nums = []
                for i in range(2, 9):
                    if i < len(parts):
                        num = int(float(parts[i]))
                        nums.append(num)
                if len(nums) == 7:
                    draws.append({
                        'period': parts[0],
                        'numbers': sorted(nums[:6]),
                        'special': nums[6]
                    })
            except (ValueError, IndexError):
                continue
    return draws


def parse_excel_file(uploaded_file) -> Optional[List[Dict]]:
    """解析Excel文件 - 精确列名匹配（修复版）"""
    try:
        # 读取Excel，强制第一行为表头
        df = pd.read_excel(uploaded_file, sheet_name=0, header=0)
        
        # 打印列名（用于调试，在终端可见）
        print(f"Excel列名: {df.columns.tolist()}")
        
        # 精确匹配列名（注意：列名可能包含空格）
        period_col = None
        date_col = None
        number_cols = []
        special_col = None
        
        for col in df.columns:
            col_str = str(col).strip()
            
            # 精确匹配期次列
            if col_str == '期次':
                period_col = col
            # 精确匹配開獎日期列
            elif col_str == '開獎日期':
                date_col = col
            # 精确匹配正码列
            elif col_str in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
                number_cols.append(col)
            # 精确匹配特码列
            elif col_str == 'B7':
                special_col = col
        
        # 如果精确匹配失败，使用位置匹配（备用方案）
        if period_col is None and len(df.columns) > 0:
            period_col = df.columns[0]  # A列是期次
            st.info(f"使用位置匹配: 期次 = {period_col}")
        
        if date_col is None and len(df.columns) > 1:
            date_col = df.columns[1]    # B列是開獎日期
            st.info(f"使用位置匹配: 開獎日期 = {date_col}")
        
        if len(number_cols) != 6 and len(df.columns) >= 8:
            number_cols = df.columns[2:8].tolist()  # C-H列是B1-B6
            st.info(f"使用位置匹配: 正码列 = {[str(c) for c in number_cols]}")
        
        if special_col is None and len(df.columns) > 8:
            special_col = df.columns[8]  # I列是B7
            st.info(f"使用位置匹配: 特码 = {special_col}")
        
        # 验证是否成功识别
        if period_col is None:
            st.error("无法识别期次列，请确保列名为'期次'")
            return None
        if date_col is None:
            st.error("无法识别開獎日期列，请确保列名为'開獎日期'")
            return None
        if len(number_cols) != 6:
            st.error(f"无法识别正码列，需要6列，找到{len(number_cols)}列")
            return None
        if special_col is None:
            st.error("无法识别特码列，请确保列名为'B7'")
            return None
        
        draws = []
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 期次
                period_val = row[period_col]
                if pd.isna(period_val):
                    continue
                period = int(period_val) if str(period_val).isdigit() else str(period_val)
                
                # 日期
                date_val = row[date_col]
                date_str = None
                if pd.notna(date_val):
                    if isinstance(date_val, datetime):
                        date_str = date_val.strftime('%Y-%m-%d')
                    elif isinstance(date_val, str):
                        # 提取日期部分
                        date_str = date_val.split()[0] if ' ' in date_val else date_val[:10]
                    else:
                        date_str = str(date_val)[:10]
                
                # 正码
                nums = []
                for col in number_cols:
                    val = row[col]
                    if pd.notna(val):
                        num = int(float(val))
                        if 1 <= num <= 49:
                            nums.append(num)
                        else:
                            raise ValueError(f"正码 {num} 超出范围1-49")
                    else:
                        raise ValueError("正码为空")
                
                if len(nums) != 6:
                    continue
                
                # 特码
                special_val = row[special_col]
                special = None
                if pd.notna(special_val):
                    special = int(float(special_val))
                    if not (1 <= special <= 49):
                        special = None
                
                draws.append({
                    'period': period,
                    'date': date_str,
                    'numbers': sorted(nums),
                    'special': special,
                    'sum': sum(nums)
                })
            except Exception as e:
                error_count += 1
                continue
        
        if error_count > 0:
            st.warning(f"跳过 {error_count} 行无效数据")
        
        # 按期次数字排序
        draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
        
        if draws:
            st.success(f"成功解析 {len(draws)} 期数据")
            return draws
        else:
            st.error("未找到有效数据")
            return None
        
    except Exception as e:
        st.error(f"Excel解析错误: {e}")
        return None

# ==================== 回测核心函数 ====================
def calculate_7code_prize(bet_numbers: List[int], draw: Dict) -> int:
    """
    计算7码复式的实际中奖金额
    bet_numbers: 7个号码的列表
    draw: 开奖数据，包含 numbers(6个正码) 和 special(特码)
    """
    draw_numbers = set(draw['numbers'])
    draw_special = draw.get('special')
    
    hit_count = sum(1 for n in bet_numbers if n in draw_numbers)
    has_special = draw_special is not None and draw_special in bet_numbers
    
    # 7码复式半注奖金（基于香港六合彩）
    if hit_count == 6:
        return 10180000  # 第1组
    elif hit_count == 5 and has_special:
        return 3060000   # 第2组
    elif hit_count == 5:
        return 61600     # 第3组
    elif hit_count == 4 and has_special:
        return 10560     # 第4组
    elif hit_count == 4:
        return 1040      # 第5组
    elif hit_count == 3 and has_special:
        return 240       # 第6组
    elif hit_count == 3:
        return 80        # 第7组
    else:
        return 0


def display_backtest_results(results_df: pd.DataFrame, num_bets: int):
    """显示回测结果"""
    st.markdown("### 📊 回测结果")
    
    stats_row = results_df[results_df['期次'] == '📊 统计']
    if len(stats_row) > 0:
        stats = stats_row.iloc[0]
        st.info(f"**{stats['日期']} | {stats['最佳匹配']} | {stats['中奖等级']} | {stats['奖金']}**")
        results_df = results_df[results_df['期次'] != '📊 统计']
    
    with st.expander("📋 查看详细回测结果"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 📈 奖金分布")
    
    prize_counts = {'第1组': 0, '第2组': 0, '第3组': 0, '第4组': 0, '第5组': 0, '第6组': 0, '第7组': 0}
    
    for _, row in results_df.iterrows():
        level = row['中奖等级']
        if level in prize_counts:
            prize_counts[level] += 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("第7组($80)", prize_counts['第7组'])
        st.metric("第6组($240)", prize_counts['第6组'])
    with col2:
        st.metric("第5组($1,040)", prize_counts['第5组'])
        st.metric("第4组($10,560)", prize_counts['第4组'])
    with col3:
        st.metric("第3组($61,600)", prize_counts['第3组'])
    with col4:
        st.metric("第2组($306万)", prize_counts['第2组'])
        st.metric("第1组($1018万)", prize_counts['第1组'])


# ==================== 管理员页面（修复版） ====================
def show_admin_page():
    """管理员页面 - 可编辑表格 + Excel上传 + 清空数据 + 回测面板"""
    
    st.subheader("📋 数据编辑器")
    st.caption("💡 双击单元格编辑 | 表格底部有 '+' 按钮添加新行")
    
    # 加载当前数据
    current_draws = load_draws_from_supabase()
    
    # 定义固定列名
    columns = ["期次", "開獎日期", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "和值"]
    
    # 构建 DataFrame（按期次数字降序显示，最新在上）
    if current_draws:
        display_draws = sorted(
            current_draws, 
            key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0, 
            reverse=True  # 降序，最新在上
        )
        data_rows = []
        for d in display_draws:
            numbers = d.get('numbers', [])
            row = {
                '期次': d.get('period', ''),
                '開獎日期': d.get('date', '')[:10] if d.get('date') else '',
                'B1': numbers[0] if len(numbers) > 0 else 0,
                'B2': numbers[1] if len(numbers) > 1 else 0,
                'B3': numbers[2] if len(numbers) > 2 else 0,
                'B4': numbers[3] if len(numbers) > 3 else 0,
                'B5': numbers[4] if len(numbers) > 4 else 0,
                'B6': numbers[5] if len(numbers) > 5 else 0,
                'B7': d.get('special', 0),
                '和值': d.get('sum', 0)
            }
            data_rows.append(row)
        df = pd.DataFrame(data_rows)
    else:
        df = pd.DataFrame(columns=columns)
    
    # 显示数据量统计
    if current_draws:
        sorted_draws = sorted(current_draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
        st.info(f"📊 当前数据量: {len(current_draws)} 期 | 范围: {sorted_draws[0].get('period')} - {sorted_draws[-1].get('period')}")
    else:
        st.info("📊 当前数据量: 0 期")
    
    # 刷新按钮
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 从数据库重新加载", use_container_width=True):
            draws = load_draws_from_supabase()
            if draws:
                st.session_state['draws_loaded'] = draws
                st.success(f"加载 {len(draws)} 期数据")
                st.rerun()
    
    st.markdown("---")
    
    # 可编辑表格
    with st.form(key="data_editor_form"):
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            key="ssq_data_editor",
            column_config={
                "期次": st.column_config.NumberColumn("期次", required=True, step=1),
                "開獎日期": st.column_config.TextColumn("開獎日期", help="格式: YYYY-MM-DD"),
                "B1": st.column_config.NumberColumn("B1", min_value=1, max_value=49, step=1),
                "B2": st.column_config.NumberColumn("B2", min_value=1, max_value=49, step=1),
                "B3": st.column_config.NumberColumn("B3", min_value=1, max_value=49, step=1),
                "B4": st.column_config.NumberColumn("B4", min_value=1, max_value=49, step=1),
                "B5": st.column_config.NumberColumn("B5", min_value=1, max_value=49, step=1),
                "B6": st.column_config.NumberColumn("B6", min_value=1, max_value=49, step=1),
                "B7": st.column_config.NumberColumn("B7", min_value=1, max_value=49, step=1),
                "和值": st.column_config.NumberColumn("和值", disabled=True),
            }
        )
        
        col_save1, col_save2, col_spacer = st.columns([1, 1, 3])
        
        with col_save1:
            overwrite_submitted = st.form_submit_button("💾 全量覆盖保存", type="primary", use_container_width=True)
        
        if overwrite_submitted:
            if edited_df is None or len(edited_df) == 0:
                st.error("没有数据可保存")
            else:
                with st.spinner("正在执行全量覆盖保存..."):
                    new_draws = []
                    errors = 0
                    
                    for idx, row in edited_df.iterrows():
                        try:
                            if pd.isna(row['期次']) or row['期次'] == 0:
                                continue
                            
                            period = int(row['期次'])
                            date = str(row['開獎日期']) if pd.notna(row['開獎日期']) and row['開獎日期'] != '' else None
                            
                            numbers = [
                                int(row['B1']) if pd.notna(row['B1']) else 0,
                                int(row['B2']) if pd.notna(row['B2']) else 0,
                                int(row['B3']) if pd.notna(row['B3']) else 0,
                                int(row['B4']) if pd.notna(row['B4']) else 0,
                                int(row['B5']) if pd.notna(row['B5']) else 0,
                                int(row['B6']) if pd.notna(row['B6']) else 0,
                            ]
                            
                            special = int(row['B7']) if pd.notna(row['B7']) else 0
                            
                            # 验证
                            valid_numbers = all(1 <= n <= 49 for n in numbers) and len(set(numbers)) == 6
                            valid_special = 1 <= special <= 49
                            
                            if valid_numbers and valid_special:
                                new_draws.append({
                                    'period': period,
                                    'date': date,
                                    'numbers': sorted(numbers),
                                    'special': special,
                                    'sum': sum(numbers)
                                })
                            else:
                                errors += 1
                        except Exception:
                            errors += 1
                    
                    if errors > 0:
                        st.warning(f"跳过 {errors} 行无效数据")
                    
                    if new_draws:
                        # 保存前排序
                        new_draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
                        if save_draws_to_supabase(new_draws):
                            st.session_state['draws_loaded'] = new_draws
                            st.success(f"全量覆盖保存 {len(new_draws)} 期数据成功！")
                            st.rerun()
                        else:
                            st.error("保存失败")
                    else:
                        st.error("没有有效数据可保存")
    
    st.markdown("---")
    
    # Excel上传区域
    st.subheader("📎 Excel文件上传")
    st.caption("格式：期次、開獎日期、B1-B6（正码）、B7（特码）")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls'],
        key="excel_uploader_admin",
        help="上传Excel文件"
    )
    
    if uploaded_file is not None:
        with st.spinner("正在解析Excel文件..."):
            excel_draws = parse_excel_file(uploaded_file)
            if excel_draws and len(excel_draws) > 0:
                # 显示数据范围
                sorted_excel = sorted(excel_draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
                st.success(f"✅ 成功解析 {len(excel_draws)} 期数据 (范围: {sorted_excel[0].get('period')} - {sorted_excel[-1].get('period')})")
                
                # 预览
                preview_data = []
                for d in excel_draws[-10:]:  # 显示最新10期
                    numbers = d.get('numbers', [])
                    numbers_str = ','.join(f"{n:02d}" for n in numbers)
                    preview_data.append({
                        '期次': d.get('period'),
                        '開獎日期': str(d.get('date', ''))[:10] if d.get('date') else '',
                        '正码': numbers_str,
                        '特码': f"{d.get('special', 0):02d}"
                    })
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
                
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ 确认全量覆盖", type="primary"):
                        excel_draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
                        if save_draws_to_supabase(excel_draws):
                            st.session_state['draws_loaded'] = excel_draws
                            st.success(f"保存 {len(excel_draws)} 期数据成功！")
                            st.rerun()
                        else:
                            st.error("保存失败")
                with col_cancel:
                    if st.button("❌ 取消"):
                        st.rerun()
            else:
                st.error("解析失败，请检查文件格式")
    
    st.markdown("---")
    
    # 清空数据
    st.subheader("🗑️ 清空所有数据")
    st.warning("⚠️ 此操作将删除Supabase中的所有历史数据，不可恢复！")
    
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("确认清空", type="secondary", key="confirm_clear"):
            supabase = init_supabase()
            if supabase:
                try:
                    supabase.schema('marksix_schema').table('marksix_draws').delete().neq("id", 0).execute()
                    st.success("✅ 数据已清空")
                    st.session_state['draws_loaded'] = []
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"清空失败: {e}")
            else:
                st.error("Supabase连接失败")
    with col_clear2:
        if st.button("取消", key="cancel_clear"):
            st.rerun()
    
    st.markdown("---")
    
    # 回测面板
    st.subheader("📊 策略回测")
    st.caption("测试不同策略在历史数据上的表现（基于当前Supabase中的数据）")
    
    draws = load_draws_from_supabase()
    
    if draws is None or len(draws) < 50:
        st.warning("⚠️ 数据不足，至少需要50期数据才能进行回测")
        return
    
    sorted_draws = sorted(draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
    st.info(f"📊 当前云端有 {len(draws)} 期数据 (范围: {sorted_draws[0].get('period')} - {sorted_draws[-1].get('period')})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_periods = st.number_input(
            "测试期数", 
            min_value=10, 
            max_value=min(200, len(draws) - 50), 
            value=min(50, len(draws) - 50), 
            step=10,
            key="backtest_periods"
        )
    with col2:
        test_bets = st.number_input("每期组数", min_value=1, max_value=20, value=4, step=1, key="backtest_bets")
    with col3:
        test_num_count = st.selectbox("每注号码数", [6, 7, 8, 9, 10], index=1, key="backtest_num_count")
    with col4:
        test_method = st.selectbox(
            "测试方法",
            ["方法1: 当前方法", "方法2: 胆拖混合", "方法3: LightGBM", "方法4: XGBoost+NN", "方法5: 综合模式"],
            index=3,
            key="backtest_method"
        )
    
    with st.expander("⚙️ 高级选项"):
        col1, col2 = st.columns(2)
        with col1:
            test_trend_window = st.number_input("趋势窗口", min_value=2, max_value=20, value=4, step=1, key="backtest_trend_window")
        with col2:
            st.markdown("**训练期数设置**")
            method1_window = st.number_input("方法1/2冷热分析期数", min_value=30, max_value=200, value=50, step=10)
            method3_window = st.number_input("方法3 LightGBM训练期数", min_value=50, max_value=300, value=100, step=10)
            method4_window = st.number_input("方法4/5 XGBoost+NN训练期数", min_value=100, max_value=500, value=200, step=20)
        
        st.markdown("**🎲 随机种子模式**")
        seed_mode_option = st.radio(
            "选择种子模式",
            options=["日期+21:15（每期用自己的开奖日期）", "用户输入固定种子", "机器自动产生（每期随机）"],
            index=0,
            key="backtest_seed_mode",
            horizontal=True
        )
        
        fixed_seed_value = 1
        if "用户输入固定种子" in seed_mode_option:
            fixed_seed_value = st.number_input("请输入固定种子值", min_value=0, max_value=10000, value=7, step=1)
    
    if st.button("▶️ 运行回测", type="primary", key="run_backtest"):
        # 映射种子模式
        if "日期+21:15" in seed_mode_option:
            seed_mode = "date"
        elif "用户输入固定种子" in seed_mode_option:
            seed_mode = "fixed"
        else:
            seed_mode = "random"
        
        # 这里调用回测函数，实际算法在第3部分实现
        st.info("回测功能将在第3部分完整实现，当前为占位界面")
        st.warning("请完成第3部分代码后，回测功能即可使用")


print("第2部分加载完成 (v7.1 - 修复排序和日期问题)")
print("=" * 60)
print("请确认第2部分代码，输入 CONFIRM 后继续第3部分")
print("=" * 60)
# ============================================================
# ============================================================
# 第3部分：5种AI算法 + 规律加权 + 可配置训练期数 + 回测实现
# 版本：v7.1
# 
# 修复内容：
#   1. 完整实现回测函数
#   2. 支持可配置训练期数
#   3. 支持3种种子模式
#   4. 规律加权已集成到方法1/2
# ============================================================

# ==================== 基础评分函数 ====================
def calculate_scores(draws: List[Dict], window_total: int = 100, window_short: int = 20, window_recent: int = 10) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    计算基础冷热码分数
    
    返回: (scores, freq, short_freq, absence)
    """
    if len(draws) < window_total:
        window_total = len(draws)
    
    total_draws = len(draws)
    expected_freq = total_draws * 6 / 49
    
    recent_draws_total = draws[-window_total:] if len(draws) >= window_total else draws
    recent_draws_short = draws[-window_short:] if len(draws) >= window_short else draws
    recent_draws_window = draws[-window_recent:] if len(draws) >= window_recent else draws
    
    # 历史频率
    freq = {i: 0 for i in range(1, 50)}
    for draw in recent_draws_total:
        for num in draw['numbers']:
            freq[num] += 1
    
    # 短期频率
    short_freq = {i: 0 for i in range(1, 50)}
    for draw in recent_draws_short:
        for num in draw['numbers']:
            short_freq[num] += 1
    expected_short = window_short * 6 / 49
    
    # 遗漏期数
    last_seen = {i: None for i in range(1, 50)}
    for idx, draw in enumerate(reversed(draws)):
        for num in draw['numbers']:
            if last_seen[num] is None:
                last_seen[num] = idx
    absence = {i: last_seen[i] if last_seen[i] is not None else total_draws for i in range(1, 50)}
    
    # 近期活跃号码
    recent_numbers = set()
    for draw in recent_draws_window:
        recent_numbers.update(draw['numbers'])
    
    # 标准化
    freq_mean = expected_freq
    freq_vals = list(freq.values())
    freq_std = np.std(freq_vals) if len(freq_vals) > 1 else 1
    
    absence_vals = list(absence.values())
    absence_mean = np.mean(absence_vals)
    absence_std = np.std(absence_vals) if len(absence_vals) > 1 else 1
    
    short_vals = list(short_freq.values())
    short_mean = expected_short
    short_std = np.std(short_vals) if len(short_vals) > 1 else 1
    
    scores = {}
    for i in range(1, 50):
        z_freq = (freq[i] - freq_mean) / freq_std if freq_std > 0 else 0
        z_absence = (absence_mean - absence[i]) / absence_std if absence_std > 0 else 0
        z_short = (short_freq[i] - short_mean) / short_std if short_std > 0 else 0
        recent_active = 1 if i in recent_numbers else -1
        
        score = 0.3 * z_freq + 0.3 * z_absence + 0.2 * z_short + 0.2 * recent_active
        scores[i] = score
    
    return scores, freq, short_freq, absence


def calculate_enhanced_scores(draws: List[Dict], window_total: int = 100, window_short: int = 20, 
                               window_recent: int = 10, zone_window: int = 20) -> Tuple[Dict[int, float], Dict[int, float], List[int]]:
    """
    计算增强分数（基础分数 + 附加分 + 规律加权）
    
    返回: (enhanced_scores, pattern_boosts, hot_zones)
    """
    base_scores, freq, short_freq, absence = calculate_scores(draws, window_total, window_short, window_recent)
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    
    # 分区热度
    zone_scores, _ = calculate_zone_heat(draws, last_n=zone_window)
    hot_zones = get_hot_zones(zone_scores, num_hot_zones=3)
    
    # 规律加权系数
    pattern_boosts = {}
    for num in range(1, 50):
        pattern_boosts[num] = calculate_six_mark_boost(num, last_numbers, last_special)
    
    # 附加分
    enhanced_scores = {}
    for num in range(1, 50):
        boost = 0.0
        
        # 上期号码加分
        if num in last_numbers:
            boost += 2.0
        if last_special and num == last_special:
            boost += 1.5
        
        # 上上期号码加分
        if len(draws) >= 2:
            prev_draw = draws[-2]
            prev_numbers = prev_draw['numbers']
            if num in prev_numbers and num not in last_numbers:
                boost += 1.0
        
        # 近期高频加分
        if len(draws) >= 5:
            last_5_draws = draws[-5:]
            count_in_last_5 = sum(1 for d in last_5_draws if num in d['numbers'])
            if count_in_last_5 >= 3:
                boost += 0.5
        
        # 热区加分
        num_zone = get_zone(num)
        if num_zone in hot_zones:
            boost += 1.2
        
        # 最终分数 = 基础分数 + 附加分，再乘以规律加权
        enhanced_scores[num] = (base_scores[num] + boost) * pattern_boosts[num]
    
    return enhanced_scores, pattern_boosts, hot_zones


# ==================== 投注生成辅助函数 ====================
def get_sampling_weights(scores: Dict[int, float], temperature: float = 1.5) -> Dict[int, float]:
    """将分数转换为采样权重"""
    weights = {}
    for num, score in scores.items():
        weights[num] = math.exp(score / temperature)
    return weights


def weighted_random_sample(weights: Dict[int, float], k: int = 7, max_attempts: int = 100) -> List[int]:
    """加权随机采样"""
    numbers = list(weights.keys())
    weight_list = [weights[n] for n in numbers]
    
    for _ in range(max_attempts):
        selected = random.choices(population=numbers, weights=weight_list, k=k)
        if len(set(selected)) == k:
            return sorted(selected)
    
    return sorted(random.sample(numbers, k))


def is_valid_combination(nums: List[int], target_sum: int, tolerance: int) -> bool:
    """验证组合是否有效（已移除连号和重复的硬性要求）"""
    total = sum(nums)
    
    if abs(total - target_sum) > tolerance:
        return False
    
    return True


def generate_one_combination(weights: Dict[int, float], num_count: int, target_sum: int, 
                              tolerance: int) -> Tuple[List[int], int]:
    """生成一组投注"""
    max_attempts = 10000
    for _ in range(max_attempts):
        selected = weighted_random_sample(weights, k=num_count)
        if is_valid_combination(selected, target_sum, tolerance):
            return selected, sum(selected)
    
    for _ in range(5000):
        selected = weighted_random_sample(weights, k=num_count)
        total = sum(selected)
        if abs(total - target_sum) <= tolerance + 5:
            return selected, total
    
    return sorted(random.sample(range(1, 50), num_count)), sum(sorted(random.sample(range(1, 50), num_count)))


# ==================== 方法1：当前方法 ====================
def generate_bets_method1_current(draws: List[Dict], num_bets: int, num_count: int,
                                    trend_window: int, random_seed: Optional[int],
                                    analysis_periods: int = 50) -> List[Dict]:
    """
    方法1：当前方法
    - 冷热码分析
    - 动态和值预测
    - 规律加权
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    # 计算增强分数（含规律加权）
    enhanced_scores, pattern_boosts, hot_zones = calculate_enhanced_scores(
        draws, window_total=analysis_periods
    )
    
    # 采样权重
    weights = get_sampling_weights(enhanced_scores, temperature=1.5)
    
    # 和值预测
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    
    bets = []
    for i in range(num_bets):
        offset = random.randint(-tolerance, tolerance)
        t = int(target_sum + offset)
        t = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, t))
        nums, total = generate_one_combination(weights, num_count, t, tolerance)
        bets.append({
            'numbers': nums,
            'sum': total,
            'target': f'方法1(目标{t})',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 方法2：胆拖混合 ====================
def select_anchor_numbers(draws: List[Dict], num_anchors: int = 3, analysis_periods: int = 50) -> List[int]:
    """
    选择胆码
    优先选择：上期号码、热区内号码、夹号-间隔2号码
    """
    enhanced_scores, pattern_boosts, hot_zones = calculate_enhanced_scores(
        draws, window_total=analysis_periods
    )
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    last_reds_sorted = sorted(last_numbers)
    
    candidates = []
    
    # 上期号码
    for num in last_numbers:
        candidates.append((num, enhanced_scores[num] + 2.0))
    if last_special:
        candidates.append((last_special, enhanced_scores[last_special] + 1.5))
    
    # 热区内号码
    for zone in hot_zones:
        for num in get_zone_numbers(zone):
            candidates.append((num, enhanced_scores[num] + 1.0))
    
    # 夹号-间隔2（最高优先级）
    for i in range(len(last_reds_sorted) - 1):
        left, right = last_reds_sorted[i], last_reds_sorted[i+1]
        if right - left == 2:
            gap_num = left + 1
            candidates.append((gap_num, enhanced_scores[gap_num] + 2.5))
    
    # 近期高频号码
    recent_20_draws = draws[-20:]
    recent_counts = {}
    for draw in recent_20_draws:
        for num in draw['numbers']:
            recent_counts[num] = recent_counts.get(num, 0) + 1
        special = draw.get('special')
        if special:
            recent_counts[special] = recent_counts.get(special, 0) + 1
    
    for num, count in recent_counts.items():
        if count >= 3:
            candidates.append((num, enhanced_scores[num] + 0.5))
    
    # 去重取最高分
    scores_dict = {}
    for num, score in candidates:
        scores_dict[num] = max(scores_dict.get(num, 0), score)
    
    sorted_candidates = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [num for num, _ in sorted_candidates[:num_anchors]]


def generate_bets_method2_hybrid(draws: List[Dict], num_bets: int, num_count: int,
                                   trend_window: int, random_seed: Optional[int],
                                   analysis_periods: int = 50) -> List[Dict]:
    """
    方法2：胆拖混合
    - 选择胆码
    - 拖码随机生成
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    # 计算增强分数
    enhanced_scores, pattern_boosts, hot_zones = calculate_enhanced_scores(
        draws, window_total=analysis_periods
    )
    
    # 采样权重
    weights = get_sampling_weights(enhanced_scores, temperature=1.5)
    
    # 和值预测
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    
    # 选择胆码
    anchors = select_anchor_numbers(draws, num_anchors=3, analysis_periods=analysis_periods)
    
    # 调整胆码权重（降低避免重复）
    for a in anchors:
        if a in weights:
            weights[a] *= 0.3
    
    bets = []
    for i in range(num_bets):
        offset = random.randint(-tolerance, tolerance)
        t = int(target_sum + offset)
        t = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, t))
        
        # 生成拖码
        remaining_needed = num_count - len(anchors)
        if remaining_needed <= 0:
            nums = sorted(anchors[:num_count])
        else:
            # 从候选池中选拖码
            candidates = [n for n in range(1, 50) if n not in anchors]
            candidate_weights = [weights.get(n, 1.0) for n in candidates]
            if sum(candidate_weights) > 0:
                candidate_weights = [w / sum(candidate_weights) for w in candidate_weights]
            else:
                candidate_weights = [1/len(candidates)] * len(candidates)
            
            selected = random.choices(candidates, weights=candidate_weights, k=remaining_needed)
            nums = sorted(anchors + selected)
        
        total = sum(nums)
        bets.append({
            'numbers': nums,
            'sum': total,
            'target': f'方法2(目标{t})',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 方法3：LightGBM ====================
def build_features_for_lightgbm(draws: List[Dict], target_num: int) -> Optional[Dict]:
    """为LightGBM构建特征（包含规律特征）"""
    if len(draws) < 20:
        return None
    
    features = {}
    total_draws = len(draws)
    
    # 基础特征
    recent_100 = draws[-100:] if len(draws) >= 100 else draws
    freq = sum(1 for d in recent_100 if target_num in d['numbers'])
    features['freq'] = freq / max(1, len(recent_100))
    
    short_draws = draws[-20:] if len(draws) >= 20 else draws
    short_freq = sum(1 for d in short_draws if target_num in d['numbers'])
    features['short_freq'] = short_freq / max(1, len(short_draws))
    
    last_seen = None
    for idx, d in enumerate(reversed(draws)):
        if target_num in d['numbers']:
            last_seen = idx
            break
    features['absence'] = last_seen if last_seen is not None else total_draws
    
    features['last_appeared'] = 1 if target_num in draws[-1]['numbers'] else 0
    features['zone'] = get_zone(target_num)
    features['recent_5'] = sum(1 for d in draws[-5:] if target_num in d['numbers'])
    features['recent_10'] = sum(1 for d in draws[-10:] if target_num in d['numbers'])
    
    # 规律特征
    last_draw = draws[-1]
    last_reds = last_draw['numbers']
    last_special = last_draw.get('special')
    all_last = last_reds + ([last_special] if last_special else [])
    
    min_edge_dist = min(abs(target_num - n) for n in all_last) if all_last else 99
    features['is_edge'] = 1 if min_edge_dist == 1 else 0
    features['min_edge_distance'] = min_edge_dist
    features['is_edge_to_special'] = 1 if last_special and abs(target_num - last_special) == 1 else 0
    
    # 夹号特征
    is_gap = 0
    gap_width = 0
    last_reds_sorted = sorted(last_reds)
    for i in range(len(last_reds_sorted) - 1):
        left, right = last_reds_sorted[i], last_reds_sorted[i+1]
        if left < target_num < right:
            is_gap = 1
            gap_width = right - left
            break
    features['is_gap'] = is_gap
    features['gap_width'] = gap_width
    
    # 连号特征
    test_reds = sorted(set(last_reds_sorted) | {target_num})
    features['consecutive_length'] = calculate_consecutive_length(test_reds, target_num)
    
    # 同尾号热度
    same_tail_count = sum(1 for d in draws[-20:] if any(n % 10 == target_num % 10 for n in d['numbers']))
    features['same_tail_hot'] = same_tail_count / max(1, min(20, len(draws)))
    
    return features


def prepare_lightgbm_dataset(draws: List[Dict], lookback: int = 100) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """准备LightGBM训练数据集"""
    if len(draws) < lookback + 10:
        return None, None
    
    X_list = []
    y_list = []
    
    for i in range(lookback, len(draws) - 1):
        train_draws = draws[i-lookback:i]
        next_draw = draws[i]
        
        for num in range(1, 50):
            features = build_features_for_lightgbm(train_draws, num)
            if features:
                X_list.append(features)
                y_list.append(1 if num in next_draw['numbers'] else 0)
    
    if not X_list:
        return None, None
    
    X_df = pd.DataFrame(X_list).fillna(0)
    y_series = pd.Series(y_list)
    
    return X_df, y_series


def train_lightgbm_model(draws: List[Dict], lookback: int = 100) -> Optional[Any]:
    """训练LightGBM模型"""
    if not LGB_AVAILABLE:
        return None
    
    X, y = prepare_lightgbm_dataset(draws, lookback=lookback)
    if X is None or len(X) < 100:
        return None
    
    try:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"LightGBM训练失败: {e}")
        return None


def predict_with_lightgbm(model: Any, draws: List[Dict]) -> Optional[List[int]]:
    """使用LightGBM预测"""
    if model is None:
        return None
    
    predictions = []
    for num in range(1, 50):
        features = build_features_for_lightgbm(draws, num)
        if features:
            X_pred = pd.DataFrame([features]).fillna(0)
            prob = model.predict_proba(X_pred)[0][1]
            predictions.append((num, prob))
        else:
            predictions.append((num, 0.0))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [num for num, prob in predictions[:7]]


def generate_bets_method3_lightgbm(draws: List[Dict], num_bets: int, num_count: int,
                                     trend_window: int, random_seed: Optional[int],
                                     lightgbm_lookback: int = 100) -> List[Dict]:
    """方法3：LightGBM（支持可配置训练期数）"""
    if not LGB_AVAILABLE:
        return generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, 50)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    model = train_lightgbm_model(draws, lookback=lightgbm_lookback)
    
    if model is None:
        return generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, 50)
    
    predicted_numbers = predict_with_lightgbm(model, draws)
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    
    bets = []
    for i in range(num_bets):
        nums = predicted_numbers[:]
        if len(nums) < num_count:
            extra = random.sample([n for n in range(1, 50) if n not in nums], num_count - len(nums))
            nums.extend(extra)
        
        replace_count = max(1, int(num_count * 0.2))
        for _ in range(replace_count):
            idx = random.randint(0, len(nums) - 1)
            new_num = random.randint(1, 49)
            while new_num in nums:
                new_num = random.randint(1, 49)
            nums[idx] = new_num
        
        nums = sorted(nums[:num_count])
        total = sum(nums)
        
        bets.append({
            'numbers': nums,
            'sum': total,
            'target': f'LightGBM(目标{target_sum})',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 方法4：XGBoost + 神经网络集成 ====================
def build_advanced_features(draws: List[Dict], target_num: int) -> Optional[Dict]:
    """构建高级特征（包含更多规律特征）"""
    if len(draws) < 30:
        return None
    
    features = {}
    total_draws = len(draws)
    
    # 基础特征
    recent_100 = draws[-100:] if len(draws) >= 100 else draws
    freq = sum(1 for d in recent_100 if target_num in d['numbers'])
    features['freq'] = freq / max(1, len(recent_100))
    
    short_draws = draws[-20:] if len(draws) >= 20 else draws
    short_freq = sum(1 for d in short_draws if target_num in d['numbers'])
    features['short_freq'] = short_freq / max(1, len(short_draws))
    
    last_seen = None
    for idx, d in enumerate(reversed(draws)):
        if target_num in d['numbers']:
            last_seen = idx
            break
    features['absence'] = last_seen if last_seen is not None else total_draws
    
    features['recent_3'] = sum(1 for d in draws[-3:] if target_num in d['numbers'])
    features['recent_5'] = sum(1 for d in draws[-5:] if target_num in d['numbers'])
    features['recent_10'] = sum(1 for d in draws[-10:] if target_num in d['numbers'])
    
    # 规律特征
    last_draw = draws[-1]
    last_reds = last_draw['numbers']
    last_special = last_draw.get('special')
    all_last = last_reds + ([last_special] if last_special else [])
    
    min_edge_dist = min(abs(target_num - n) for n in all_last) if all_last else 99
    features['is_edge'] = 1 if min_edge_dist == 1 else 0
    features['min_edge_distance'] = min_edge_dist
    features['edge_count'] = sum(1 for n in all_last if abs(target_num - n) == 1)
    
    # 夹号特征
    is_gap = 0
    gap_width = 0
    gap_count = 0
    last_reds_sorted = sorted(last_reds)
    for i in range(len(last_reds_sorted) - 1):
        left, right = last_reds_sorted[i], last_reds_sorted[i+1]
        if left < target_num < right:
            is_gap = 1
            gap_width = right - left
            gap_count += 1
    features['is_gap'] = is_gap
    features['gap_width'] = gap_width
    features['gap_count'] = gap_count
    features['is_gap_2'] = 1 if (is_gap and gap_width == 2) else 0
    
    # 连号特征
    test_reds = sorted(set(last_reds_sorted) | {target_num})
    features['consecutive_length'] = calculate_consecutive_length(test_reds, target_num)
    
    # 分区特征
    features['zone'] = get_zone(target_num)
    zone_scores, _ = calculate_zone_heat(draws)
    features['is_hot_zone'] = 1 if get_zone(target_num) in get_hot_zones(zone_scores) else 0
    
    # 同尾号
    same_tail_nums = [n for n in range(1, 50) if n % 10 == target_num % 10]
    same_tail_freq = sum(1 for d in draws[-20:] for n in d['numbers'] if n in same_tail_nums)
    features['same_tail_hot'] = same_tail_freq / max(1, 20 * len(same_tail_nums))
    
    # 与特码关系
    if last_special:
        features['diff_to_special'] = abs(target_num - last_special)
        features['is_special'] = 1 if target_num == last_special else 0
    
    return features


def prepare_advanced_dataset(draws: List[Dict], lookback: int = 200) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """准备高级数据集"""
    if len(draws) < lookback + 10:
        return None, None
    
    X_list = []
    y_list = []
    
    for i in range(lookback, len(draws) - 1):
        train_draws = draws[i-lookback:i]
        next_draw = draws[i]
        
        for num in range(1, 50):
            features = build_advanced_features(train_draws, num)
            if features:
                X_list.append(features)
                y_list.append(1 if num in next_draw['numbers'] else 0)
    
    if not X_list:
        return None, None
    
    X_df = pd.DataFrame(X_list).fillna(0)
    y_series = pd.Series(y_list)
    
    return X_df, y_series


def train_xgboost_nn_ensemble(draws: List[Dict], lookback: int = 200) -> Optional[Dict]:
    """训练XGBoost + 神经网络集成"""
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    
    X, y = prepare_advanced_dataset(draws, lookback=lookback)
    if X is None or len(X) < 200:
        return None
    
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        xgb_model.fit(X, y)
        nn_model.fit(X_scaled, y)
        
        return {
            'xgb': xgb_model,
            'nn': nn_model,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
    except Exception as e:
        print(f"XGBoost+NN集成训练失败: {e}")
        return None


def predict_with_ensemble(model_dict: Dict, draws: List[Dict]) -> Optional[List[int]]:
    """使用集成模型预测"""
    if model_dict is None:
        return None
    
    try:
        predictions = []
        for num in range(1, 50):
            features = build_advanced_features(draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                
                missing_cols = set(model_dict['feature_names']) - set(X_pred.columns)
                for col in missing_cols:
                    X_pred[col] = 0
                X_pred = X_pred[model_dict['feature_names']]
                
                xgb_prob = model_dict['xgb'].predict_proba(X_pred)[0][1]
                
                X_scaled = model_dict['scaler'].transform(X_pred)
                nn_prob = model_dict['nn'].predict_proba(X_scaled)[0][1]
                
                ensemble_prob = xgb_prob * 0.5 + nn_prob * 0.5
                predictions.append((num, ensemble_prob))
            else:
                predictions.append((num, 0.0))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, prob in predictions[:7]]
    except Exception as e:
        print(f"集成预测失败: {e}")
        return None


def generate_bets_method4_ensemble(draws: List[Dict], num_bets: int, num_count: int,
                                     trend_window: int, random_seed: Optional[int],
                                     ensemble_lookback: int = 200) -> List[Dict]:
    """方法4：XGBoost + 神经网络集成（支持可配置训练期数）"""
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, 100)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    ensemble = train_xgboost_nn_ensemble(draws, lookback=ensemble_lookback)
    
    if ensemble is None:
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, 100)
    
    predicted_numbers = predict_with_ensemble(ensemble, draws)
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    
    bets = []
    for i in range(num_bets):
        nums = predicted_numbers[:]
        if len(nums) < num_count:
            extra = random.sample([n for n in range(1, 50) if n not in nums], num_count - len(nums))
            nums.extend(extra)
        
        replace_count = max(1, int(num_count * 0.1))
        for _ in range(replace_count):
            idx = random.randint(0, len(nums) - 1)
            new_num = random.randint(1, 49)
            while new_num in nums:
                new_num = random.randint(1, 49)
            nums[idx] = new_num
        
        nums = sorted(nums[:num_count])
        total = sum(nums)
        
        bets.append({
            'numbers': nums,
            'sum': total,
            'target': f'XGBoost+NN(目标{target_sum})',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 方法5：综合模式 ====================
def generate_bets_method5_ensemble(draws: List[Dict], num_bets: int, num_count: int,
                                     trend_window: int, random_seed: Optional[int],
                                     method1_window: int = 50, method2_window: int = 50,
                                     method3_window: int = 100, method4_window: int = 200) -> List[Dict]:
    """方法5：综合模式（运行方法1-4，取高频号码 + 规律加权）"""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    all_numbers = []
    
    # 方法1
    bets1 = generate_bets_method1_current(draws, num_bets, num_count, trend_window, random_seed, method1_window)
    for bet in bets1:
        all_numbers.extend(bet['numbers'])
    
    # 方法2
    bets2 = generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, method2_window)
    for bet in bets2:
        all_numbers.extend(bet['numbers'])
    
    # 方法3
    bets3 = generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, method3_window)
    for bet in bets3:
        all_numbers.extend(bet['numbers'])
    
    # 方法4
    bets4 = generate_bets_method4_ensemble(draws, num_bets, num_count, trend_window, random_seed, method4_window)
    for bet in bets4:
        all_numbers.extend(bet['numbers'])
    
    # 统计频率
    freq_counter = Counter(all_numbers)
    
    # 规律加权
    last_draw = draws[-1]
    last_reds = last_draw['numbers']
    last_special = last_draw.get('special')
    
    weighted_scores = {}
    for num in range(1, 50):
        base_freq = freq_counter.get(num, 0)
        pattern_boost = calculate_six_mark_boost(num, last_reds, last_special)
        weighted_scores[num] = base_freq * pattern_boost
    
    # 取Top N
    top_numbers = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:num_count]
    final_numbers = sorted([num for num, _ in top_numbers])
    
    # 生成多组（通过微调）
    bets = []
    for i in range(num_bets):
        nums = final_numbers.copy()
        
        if i > 0:
            replace_count = min(i, 2)
            for _ in range(replace_count):
                idx = random.randint(0, len(nums) - 1)
                candidates = [n for n in range(1, 50) if n not in nums]
                if candidates:
                    nums[idx] = random.choice(candidates)
            nums = sorted(nums)
        
        total = sum(nums)
        base_target = get_target_sum_by_numbers_count(num_count)
        
        bets.append({
            'numbers': nums,
            'sum': total,
            'target': '方法5:综合模式',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 回测函数（完整实现） ====================
def run_backtest(draws: List[Dict], method_name: str, num_bets: int, num_count: int,
                  trend_window: int, test_periods: int,
                  method1_window: int, method2_window: int, 
                  method3_window: int, method4_window: int,
                  seed_mode: str = "date", fixed_seed_value: int = 1) -> Optional[pd.DataFrame]:
    """
    运行回测（完整实现）
    支持3种种子模式
    """
    # 确定训练窗口
    if "方法1" in method_name or "方法2" in method_name:
        train_window = method1_window
    elif "方法3" in method_name:
        train_window = method3_window
    else:
        train_window = method4_window
    
    if len(draws) < train_window + test_periods:
        st.error(f"数据不足：需要至少 {train_window + test_periods} 期数据")
        return None
    
    # 方法偏移量
    method_seed_offset = {
        "方法1": 100, "方法2": 200, "方法3": 300, "方法4": 400, "方法5": 500
    }.get(method_name.split(":")[0], 0)
    
    results = []
    total_cost = 0
    total_prize = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(test_periods):
        # 训练数据：使用当期之前的数据
        train_draws = draws[:-(test_periods - i)]
        test_draw = draws[-(test_periods - i)]
        
        # 设置随机种子
        if seed_mode == "date":
            test_date = test_draw.get('date')
            if test_date:
                try:
                    dt = datetime.strptime(test_date[:10], '%Y-%m-%d')
                    seed_val = int(datetime(dt.year, dt.month, dt.day, 21, 15).timestamp())
                    seed_val += method_seed_offset
                except:
                    seed_val = 42 + method_seed_offset + i
            else:
                seed_val = 42 + method_seed_offset + i
        elif seed_mode == "fixed":
            seed_val = fixed_seed_value + method_seed_offset
        else:
            seed_val = random.randint(0, 1000000) + method_seed_offset
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        status_text.text(f"正在回测第 {i+1}/{test_periods} 期...")
        progress_bar.progress((i + 1) / test_periods)
        
        # 根据方法生成投注
        if "方法1" in method_name:
            bets = generate_bets_method1_current(
                train_draws, num_bets, num_count, trend_window, seed_val, method1_window
            )
        elif "方法2" in method_name:
            bets = generate_bets_method2_hybrid(
                train_draws, num_bets, num_count, trend_window, seed_val, method2_window
            )
        elif "方法3" in method_name:
            bets = generate_bets_method3_lightgbm(
                train_draws, num_bets, num_count, trend_window, seed_val, method3_window
            )
        elif "方法4" in method_name:
            bets = generate_bets_method4_ensemble(
                train_draws, num_bets, num_count, trend_window, seed_val, method4_window
            )
        else:
            bets = generate_bets_method5_ensemble(
                train_draws, num_bets, num_count, trend_window, seed_val,
                method1_window, method2_window, method3_window, method4_window
            )
        
        # 计算每期最佳匹配和奖金
        best_match = 0
        best_special_match = False
        best_prize_amount = 0
        best_prize_desc = "无中奖"
        
        for bet in bets:
            prize_amount = calculate_7code_prize(bet['numbers'], test_draw)
            match_count = len(set(bet['numbers'][:6]) & set(test_draw['numbers']))
            special_match = test_draw.get('special') in bet['numbers'] if test_draw.get('special') else False
            
            if match_count > best_match or (match_count == best_match and special_match):
                best_match = match_count
                best_special_match = special_match
                best_prize_amount = prize_amount
                if prize_amount >= 10000000:
                    best_prize_desc = "第1组"
                elif prize_amount >= 3000000:
                    best_prize_desc = "第2组"
                elif prize_amount >= 60000:
                    best_prize_desc = "第3组"
                elif prize_amount >= 9600:
                    best_prize_desc = "第4组"
                elif prize_amount >= 640:
                    best_prize_desc = "第5组"
                elif prize_amount >= 320:
                    best_prize_desc = "第6组"
                elif prize_amount >= 40:
                    best_prize_desc = "第7组"
        
        total_cost += num_bets * 35
        total_prize += best_prize_amount
        
        match_display = f"{best_match}+特" if best_special_match and best_match >= 3 else str(best_match) if best_match >= 3 else str(best_match)
        
        results.append({
            '期次': test_draw.get('period', ''),
            '日期': test_draw.get('date', '')[:10] if test_draw.get('date') else '',
            '最佳匹配': match_display if best_match >= 3 else "-",
            '中奖等级': best_prize_desc,
            '奖金': f"${best_prize_amount:,.0f}" if best_prize_amount > 0 else "-"
        })
    
    progress_bar.empty()
    status_text.empty()
    
    roi = (total_prize - total_cost) / total_cost * 100 if total_cost > 0 else 0
    results.append({
        '期次': '📊 统计',
        '日期': f'测试期数: {test_periods}',
        '最佳匹配': f'总投入: ${total_cost:,.0f}',
        '中奖等级': f'总奖金: ${total_prize:,.0f}',
        '奖金': f'ROI: {roi:+.1f}%'
    })
    
    return pd.DataFrame(results)


print("第3部分加载完成 (v7.1 - 完整回测实现)")
print("=" * 60)
print("请确认第3部分代码，输入 CONFIRM 后继续第4部分")
print("=" * 60)
# ============================================================
# ============================================================
# 第4部分：主页面UI + 智能投注生成 + 侧边栏
# 版本：v7.1
# 
# 修复内容：
#   1. 使用 get_latest_and_oldest() 正确获取最新/最早期次
#   2. 修复数据概览显示
#   3. 完整集成回测功能
# ============================================================

# ==================== 初始化Session State ====================
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_admin' not in st.session_state:
    st.session_state['show_admin'] = False
if 'preview_draws' not in st.session_state:
    st.session_state['preview_draws'] = None
if 'generated_bets' not in st.session_state:
    st.session_state['generated_bets'] = None
if 'model_used' not in st.session_state:
    st.session_state['model_used'] = None
if 'draws_loaded' not in st.session_state:
    st.session_state['draws_loaded'] = None

# ==================== 右上角齿轮图标 ====================
col_title, col_settings = st.columns([0.95, 0.05])
with col_settings:
    if st.button("⚙️", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

if st.session_state.get('show_admin', False):
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        show_admin_page()
        admin_logout()

# ==================== 加载数据 ====================
if st.session_state.get('draws_loaded') is None:
    with st.spinner("加载数据中..."):
        draws = load_recent_draws_from_supabase(limit=300)
        if draws:
            st.session_state['draws_loaded'] = draws
        else:
            st.session_state['draws_loaded'] = []

draws = st.session_state.get('draws_loaded', [])

if not draws or len(draws) < 5:
    st.info("👈 请点击右上角齿轮图标，进入管理员页面导入历史数据")
    st.stop()

# ==================== 主页面标题 ====================
st.title("🎯 六合彩AI智能选号工具 - 云端ML完整版 v7.1")

# ==================== 显示数据概览（修复版） ====================
st.subheader("📊 数据概览")

# 按期次数字排序获取最新和最旧
sorted_draws = sorted(draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
latest_draw = sorted_draws[-1] if sorted_draws else {}
oldest_draw = sorted_draws[0] if sorted_draws else {}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("最新期次", latest_draw.get('period', 'N/A'))
with col2:
    latest_date = latest_draw.get('date', 'N/A')
    st.metric("最新日期", latest_date[:10] if latest_date and len(latest_date) > 10 else latest_date)
with col3:
    st.metric("最早期次", oldest_draw.get('period', 'N/A'))
with col4:
    st.metric("数据总量", f"{len(draws)} 期")

st.markdown("---")

# ==================== 冷热码分析 ====================
st.subheader("🔥 冷热码分析")

col1, col2 = st.columns(2)
with col1:
    analysis_periods = st.number_input(
        "分析期数", 
        min_value=10, 
        max_value=min(300, len(draws)), 
        value=min(100, len(draws)), 
        step=10,
        key="analysis_periods"
    )
with col2:
    zone_periods = st.number_input(
        "分区分析期数",
        min_value=10,
        max_value=min(300, len(draws)),
        value=min(50, len(draws)),
        step=10,
        key="zone_periods"
    )

# 计算分数
enhanced_scores, pattern_boosts, hot_zones = calculate_enhanced_scores(
    draws, window_total=analysis_periods
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔥 热门号码 (Top 15)**")
    hot_df = pd.DataFrame([
        {'号码': num, '得分': f"{enhanced_scores[num]:.2f}", '规律加权': f"{pattern_boosts[num]:.2f}x"}
        for num in sorted(enhanced_scores, key=enhanced_scores.get, reverse=True)[:15]
    ])
    st.dataframe(hot_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**❄️ 冷门号码 (Bottom 10)**")
    cold_df = pd.DataFrame([
        {'号码': num, '得分': f"{enhanced_scores[num]:.2f}"}
        for num in sorted(enhanced_scores, key=enhanced_scores.get)[:10]
    ])
    st.dataframe(cold_df, use_container_width=True, hide_index=True)

with col3:
    st.markdown("**🔥 当前热区 (7分区)**")
    zone_scores, zone_hits = calculate_zone_heat(draws, last_n=zone_periods)
    hot_zones_list = get_hot_zones(zone_scores, num_hot_zones=3)
    
    zone_display = []
    for z in range(1, 8):
        zone_range = f"{get_zone_numbers(z)[0]:02d}-{get_zone_numbers(z)[-1]:02d}"
        is_hot = z in hot_zones_list
        zone_display.append({
            '分区': f"{chr(64+z)}区 ({zone_range})",
            '热度': '🔥' if is_hot else '❄️',
            '出现次数': zone_hits[z]
        })
    st.dataframe(pd.DataFrame(zone_display), use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== 和值趋势分析 ====================
st.subheader("📈 和值趋势分析")

show_periods = st.slider(
    "显示最近期数", 
    min_value=10, 
    max_value=min(200, len(draws)), 
    value=min(100, len(draws)), 
    step=10
)

recent_sum_draws = draws[-show_periods:] if len(draws) >= show_periods else draws
sum_7_values = [convert_6sum_to_7sum(draw['sum']) for draw in recent_sum_draws]
sum_df = pd.DataFrame([{'期次': i+1, '和值(7码)': val} for i, val in enumerate(sum_7_values)])

fig = px.line(sum_df, x='期次', y='和值(7码)', title=f'最近{show_periods}期和值走势')
fig.add_hline(y=175, line_dash="dash", line_color="red", annotation_text="理论均值(175)")
fig.add_hrect(y0=140, y1=210, line_width=0, fillcolor="green", opacity=0.1, annotation_text="约68%区间")
st.plotly_chart(fig, use_container_width=True)

# ==================== 智能投注生成 ====================
st.subheader("🎲 智能投注生成")

next_period = get_next_period(draws)
st.info(f"🎯 **预测下一期**: {next_period}")

col1, col2, col3 = st.columns(3)
with col1:
    num_bets = st.number_input("购买组数", min_value=1, max_value=100, value=4, step=1, key="num_bets")
with col2:
    num_count = st.selectbox("每注号码个数", [6, 7, 8, 9, 10], index=1, key="num_count")
with col3:
    ai_model = st.selectbox(
        "🤖 AI预测模型",
        [
            "方法1: 当前方法",
            "方法2: 胆拖混合",
            "方法3: LightGBM",
            "方法4: XGBoost+NN",
            "方法5: 综合模式 ⭐推荐"
        ],
        index=4,
        key="ai_model"
    )

with st.expander("⚙️ 高级设置"):
    col1, col2 = st.columns(2)
    with col1:
        trend_window = st.number_input("和值趋势窗口", min_value=2, max_value=20, value=4, step=1, key="trend_window")
    with col2:
        seed_input = st.text_input("随机种子", value="", placeholder="留空使用系统时间 | 或输入日期如: 2026-05-13", key="seed_input")
    
    st.markdown("**📊 训练期数设置**")
    col1, col2, col3 = st.columns(3)
    with col1:
        method1_window = st.number_input("方法1/2期数", min_value=30, max_value=200, value=50, step=10, key="m1_window")
    with col2:
        method3_window = st.number_input("方法3 LightGBM期数", min_value=50, max_value=300, value=100, step=10, key="m3_window")
    with col3:
        method4_window = st.number_input("方法4/5 XGBoost+NN期数", min_value=100, max_value=500, value=200, step=20, key="m4_window")

# 显示和值预测信息
target_sum, tolerance, direction, direction_desc, mean_sum, std_sum, short_mean = get_dynamic_sum_range(
    draws, num_count, window=trend_window
)
st.caption(f"💡 **和值动态预测**: 长期均值={mean_sum:.1f}, σ={std_sum:.1f} | {direction_desc} | 目标={target_sum} | 容差=±{tolerance}")

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    # 解析随机种子
    random_seed = None
    if seed_input and seed_input.strip():
        random_seed = parse_datetime_string(seed_input)
        if random_seed:
            st.success(f"✅ 已设置随机种子: {seed_input}")
        else:
            st.warning(f"⚠️ 无法解析 '{seed_input}'，将使用系统时间")
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        if "方法1" in ai_model:
            bets = generate_bets_method1_current(
                draws, num_bets, num_count, trend_window, random_seed, method1_window
            )
            model_used = "方法1: 当前方法"
        elif "方法2" in ai_model:
            bets = generate_bets_method2_hybrid(
                draws, num_bets, num_count, trend_window, random_seed, method1_window
            )
            model_used = "方法2: 胆拖混合"
        elif "方法3" in ai_model:
            bets = generate_bets_method3_lightgbm(
                draws, num_bets, num_count, trend_window, random_seed, method3_window
            )
            model_used = "方法3: LightGBM"
        elif "方法4" in ai_model:
            bets = generate_bets_method4_ensemble(
                draws, num_bets, num_count, trend_window, random_seed, method4_window
            )
            model_used = "方法4: XGBoost+NN"
        else:
            bets = generate_bets_method5_ensemble(
                draws, num_bets, num_count, trend_window, random_seed,
                method1_window, method1_window, method3_window, method4_window
            )
            model_used = "方法5: 综合模式"
    
    st.session_state['generated_bets'] = bets
    st.session_state['model_used'] = model_used
    st.success(f"✅ 使用 {model_used} 生成 {len(bets)} 组{num_count}码复式")

# 显示生成的投注
if st.session_state['generated_bets'] is not None:
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知模型')
    num_count_display = len(bets[0]['numbers']) if bets else 7
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    st.caption(f"{num_count_display}码复式 = {num_count_display}注（每注$5半注），每组成本${num_count_display * 5}")
    
    bets_data = []
    for i, bet in enumerate(bets, 1):
        numbers_display = ','.join(f"{n:02d}" for n in bet['numbers'])
        bets_data.append({
            '组别': i,
            f'{num_count_display}个号码': numbers_display,
            '和值': bet['sum'],
            '策略': bet['target'],
            '偏差': f"{bet['deviation']:+d}"
        })
    
    st.dataframe(pd.DataFrame(bets_data), use_container_width=True, hide_index=True)

# ==================== 多期查奖 ====================
st.markdown("---")
st.markdown("### 🔍 多期查奖")
st.caption("📌 粘贴实际开奖数据，查看每组复式中奖情况（最多5期）")

check_draws_text = st.text_area(
    "📋 粘贴多期开奖数据（最多5期）",
    height=120,
    key="check_draws",
    placeholder="示例:\n26045 2026-04-25 4 16 21 36 42 46 9\n26044 2026-04-23 12 23 37 38 45 48 8"
)

def calculate_match_score_for_draws(bet_numbers: List[int], check_draws: List[Dict]) -> List[str]:
    """计算多期中奖分数"""
    results = []
    for draw in check_draws:
        draw_numbers = set(draw['numbers'])
        draw_special = draw.get('special')
        
        main_matches = len(set(bet_numbers[:6]) & draw_numbers)
        special_match = (len(bet_numbers) >= 7 and bet_numbers[6] == draw_special) if draw_special else False
        
        if special_match:
            score = main_matches + 0.5
        else:
            score = float(main_matches)
        
        if score == int(score):
            results.append(str(int(score)))
        else:
            results.append(f"{score:.1f}")
    
    return results

if st.button("🔍 查奖", key="check_btn") and check_draws_text:
    check_draws = parse_multi_draws_for_checking(check_draws_text, max_draws=5)
    if check_draws:
        st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
        
        if st.session_state.get('generated_bets'):
            enhanced_bets_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                numbers_display = ','.join(f"{n:02d}" for n in bet['numbers'])
                row = {'组别': i, f'{len(bet["numbers"])}个号码': numbers_display, '和值': bet['sum']}
                
                match_scores = calculate_match_score_for_draws(bet['numbers'], check_draws)
                for idx, draw in enumerate(check_draws):
                    period_str = str(draw['period'])
                    if len(period_str) > 10:
                        period_str = period_str[-10:]
                    row[f'中奖_{period_str}'] = match_scores[idx]
                
                enhanced_bets_data.append(row)
            
            st.dataframe(pd.DataFrame(enhanced_bets_data), use_container_width=True, hide_index=True)
            
            # 显示开奖数据预览
            preview_df = pd.DataFrame([
                {'期次': d['period'], '正码': str(d['numbers']), '特码': d['special']}
                for d in check_draws
            ])
            st.markdown("**📊 开奖数据预览**")
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.warning("请先生成投注组合")
    else:
        st.error("解析失败，请检查格式")

# ==================== 底部 ====================
st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。六合彩本质上是一种随机游戏，长期期望值为负，请理性投注。")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🎰 六合彩AI分析工具 v7.1")
    st.markdown("---")
    
    with st.expander("🤖 ML库状态", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if LGB_AVAILABLE:
                st.success("✅ LightGBM")
            else:
                st.error("❌ LightGBM")
            if XGB_AVAILABLE:
                st.success("✅ XGBoost")
            else:
                st.error("❌ XGBoost")
        with col2:
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn")
            else:
                st.error("❌ scikit-learn")
    
    with st.expander("📖 五种AI算法对比"):
        st.markdown("""
        | 算法 | 核心原理 | 规律加权 |
        |------|---------|---------|
        | 🟢 方法1 | 冷热码+和值 | ✅ 已集成 |
        | 🟡 方法2 | 胆拖混合 | ✅ 已集成 |
        | 🔵 方法3 | LightGBM | 规律特征 |
        | 🟣 方法4 | XGBoost+NN | 规律特征 |
        | 🌟 方法5 | 综合投票 | ✅ 已集成 |
        """)
    
    with st.expander("💰 奖金结构（7码复式半注）"):
        st.markdown("""
        | 等级 | 条件 | 总奖金 |
        |------|------|--------|
        | 第7组 | 中3码 | $80 |
        | 第6组 | 中3+特 | $240 |
        | 第5组 | 中4码 | $1,040 |
        | 第4组 | 中4+特 | $10,560 |
        | 第3组 | 中5码 | ~$61,600 |
        | 第2组 | 中5+特 | ~$3,060,000 |
        | 第1组 | 中6码 | ~$10,180,000 |
        """)
    
    with st.expander("📐 规律加权说明"):
        st.markdown("""
        | 规律 | 加权系数 |
        |------|---------|
        | 边号(正码) | ×1.3 |
        | 边号(特码) | ×1.2 |
        | 夹号-间隔2 | ×2.5 |
        | 夹号-间隔3 | ×1.8 |
        | 夹号-间隔4 | ×1.3 |
        | 连号-2连 | ×1.2 |
        | 连号-3连+ | ×1.5 |
        | 重号(特码) | ×1.15 |
        """)
        st.caption("基于269期历史数据验证")
    
    st.markdown("---")
    st.caption("DFSS智能选号工具 v7.1")
    st.caption("更新: 2026-05-13")
    st.caption("修复: 数据排序 | 日期格式 | 回测完整实现")


print("第4部分加载完成 (v7.1 - 修复版)")
print("=" * 60)
print("所有代码加载完成！六合彩AI智能选号工具 v7.1 已就绪。")
print("=" * 60)

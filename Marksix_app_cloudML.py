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
from math import comb
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
import requests
from bs4 import BeautifulSoup

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
    """保存开奖数据到Supabase（覆盖保存）- 保存7码和值"""
    supabase = init_supabase()
    if supabase is None:
        return False
    try:
        # 清空现有数据
        supabase.schema('marksix_schema').table('marksix_draws').delete().neq("id", 0).execute()
        
        for draw in draws:
            numbers = draw['numbers']  # 6个正码
            special = draw.get('special', 0)
            sum_7 = sum(numbers) + special  # 7码和值 = 6码和值 + 特码
            
            data = {
                "period": draw.get('period'),
                "date": draw.get('date'),
                "numbers": numbers,
                "special": draw.get('special', 0),
                "sum_value": sum_7,  # 也存入 sum_value 保持兼容
                "sum_7": sum_7       # 新增 sum_7 字段
            }
            supabase.schema('marksix_schema').table('marksix_draws').insert(data).execute()
        
        return True
    except Exception as e:
        st.error(f"保存到Supabase失败: {e}")
        return False
#------------
def incremental_sync_draws(draws: List[Dict]) -> Dict:
    """增量同步：只更新变更的数据（优化版）"""
    if not draws:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    supabase = init_supabase()
    if supabase is None:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    try:
        # ========== 统一期次为数字 ==========
        normalized_draws = []
        for draw in draws:
            if draw.get('period') is None:
                continue
            normalized_draw = draw.copy()
            # 期次统一转为整数（六合彩期次是数字）
            if isinstance(draw['period'], str) and draw['period'].isdigit():
                normalized_draw['period'] = int(draw['period'])
            normalized_draws.append(normalized_draw)
        
        # 获取数据库中现有期次
        existing_response = supabase.schema('marksix_schema').table('marksix_draws')\
            .select("period").execute()
        existing_periods = {row["period"] for row in existing_response.data} if existing_response.data else set()
        
        # 新数据中的期次
        new_periods = {draw['period'] for draw in normalized_draws if draw.get('period') is not None}
        
        # 需要删除的期次
        to_delete = existing_periods - new_periods
        
        # 需要新增的期次
        to_insert = new_periods - existing_periods
        
        inserted = 0
        updated = 0
        deleted = 0
        
        # 执行删除
        for period in to_delete:
            try:
                supabase.schema('marksix_schema').table('marksix_draws')\
                    .delete().eq("period", period).execute()
                deleted += 1
            except Exception as e:
                st.warning(f"删除期次 {period} 失败: {e}")
        
        # 准备需要 upsert 的数据
        upsert_data = []
        for draw in normalized_draws:
            period = draw['period']
            numbers = draw.get('numbers', [])
            upsert_data.append({
                "period": period,
                "date": draw.get('date'),
                "numbers": numbers,
                "special": draw.get('special', 0),
                "sum_value": sum(numbers),  # 7码和值
                "sum_7": sum(numbers)       # 新增 sum_7 字段
            })
        
        # 使用 upsert 批量操作
        if upsert_data:
            result = supabase.schema('marksix_schema').table('marksix_draws')\
                .upsert(upsert_data, on_conflict='period').execute()
            inserted = len(to_insert)
            updated = len(new_periods) - len(to_insert)
        
        return {"inserted": inserted, "updated": updated, "deleted": deleted}
        
    except Exception as e:
        st.error(f"增量同步失败: {e}")
        return {"inserted": 0, "updated": 0, "deleted": 0}
#-------------
def load_recent_draws_from_supabase(limit: int = 500) -> Optional[List[Dict]]:
    """加载最近N期数据（按期次降序取N条，再反转）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    try:
        response = supabase.schema('marksix_schema').table('marksix_draws')\
            .select("*")\
            .order("period", desc=True)\
            .limit(limit)\
            .execute()
        
        if not response.data:
            return None
        
        draws = []
        for row in reversed(response.data):
            # 优先使用 sum_7，如果没有则兼容旧数据
            sum_7 = row.get('sum_7', row.get('sum_value', 0))
            if sum_7 == 0 and row.get('numbers'):
                numbers = row.get('numbers', [])
                special = row.get('special', 0)
                sum_7 = sum(numbers) + special
            
            draws.append({
                'period': row.get('period'),
                'date': row.get('date'),
                'numbers': row['numbers'],
                'special': row.get('special'),
                'sum': sum_7  # 7码和值
            })
        
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None
#-------------
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
#------
def load_draws_from_supabase() -> Optional[List[Dict]]:
    """从Supabase加载全部开奖数据"""
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
            # 优先使用 sum_7
            sum_7 = row.get('sum_7', row.get('sum_value', 0))
            if sum_7 == 0 and row.get('numbers'):
                numbers = row.get('numbers', [])
                special = row.get('special', 0)
                sum_7 = sum(numbers) + special
            
            draws.append({
                'period': row.get('period'),
                'date': row.get('date'),
                'numbers': row['numbers'],
                'special': row.get('special'),
                'sum': sum_7
            })
        
        # 按期次排序
        draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None


# ==================== 辅助函数 ====================
def convert_6sum_to_7sum(sum_6: int) -> int:
    """将6码和值转换为7码和值（兼容旧数据）"""
    # 如果已经是7码和值（约140-210），直接返回
    if 140 <= sum_6 <= 210:
        return sum_6
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

#----------------
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

# ==================== 移动平均和值预测（7期±17）- 7码和值 ====================

def get_target_sum_moving_average(draws: List[Dict], window: int = 7, tolerance: int = 17) -> Tuple[int, int]:
    """
    移动平均和值预测（7码和值）
    
    返回:
        (中心值, 容差)
    """
    if len(draws) < window:
        return 175, tolerance
    
    recent_sums = []
    for draw in draws[-window:]:
        numbers = draw.get('numbers', [])
        special = draw.get('special', 0)
        sum_7 = sum(numbers) + special  # 7码和值
        recent_sums.append(sum_7)
    
    if not recent_sums:
        return 175, tolerance
    
    mean_val = np.mean(recent_sums)
    return int(mean_val), tolerance


def get_target_sum_moving_average_range(draws: List[Dict], window: int = 7, tolerance: int = 17) -> Tuple[int, int]:
    """移动平均和值范围预测（7码和值）"""
    center, tol = get_target_sum_moving_average(draws, window, tolerance)
    lower = max(140, center - tol)  # 7码最低140
    upper = min(210, center + tol)  # 7码最高210
    return lower, upper


def generate_target_sum_by_moving_average(draws: List[Dict]) -> int:
    """在移动平均预测范围内随机生成一个和值"""
    lower, upper = get_target_sum_moving_average_range(draws)
    return random.randint(lower, upper)


# ==================== 正弦拟合和值预测（±17） ====================

# ==================== 正弦拟合和值预测（±17）- 7码和值 ====================

def sine_fit_predict_sum_marksix(recent_sums: List[int]) -> int:
    """
    正弦拟合预测下一期和值（7码和值）
    """
    from scipy.optimize import curve_fit
    
    if len(recent_sums) < 10:
        return int(round(np.mean(recent_sums))) if recent_sums else 175
    
    def sine_func(x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C
    
    x = np.arange(len(recent_sums))
    y = np.array(recent_sums)
    
    A_guess = (np.max(y) - np.min(y)) / 2
    C_guess = np.mean(y)
    omega_guess = 2 * np.pi / 6.5
    
    try:
        params, _ = curve_fit(
            sine_func, x, y,
            p0=[A_guess, omega_guess, 0, C_guess],
            maxfev=2000
        )
        A, omega, phi, C = params
        next_val = sine_func(len(recent_sums), A, omega, phi, C)
        return max(140, min(210, int(round(next_val))))
    except:
        return int(round(np.mean(recent_sums)))


def get_target_sum_sine_range(draws: List[Dict], tolerance: int = 17) -> Tuple[int, int]:
    """正弦拟合和值范围预测（7码和值）"""
    if len(draws) < 10:
        return 158, 192  # 7码范围
    
    recent_sums = []
    for draw in draws[-10:]:
        numbers = draw.get('numbers', [])
        special = draw.get('special', 0)
        sum_7 = sum(numbers) + special  # 7码和值
        recent_sums.append(sum_7)
    
    if len(recent_sums) < 10:
        return 158, 192
    
    target = sine_fit_predict_sum_marksix(recent_sums)
    lower = max(140, target - tolerance)
    upper = min(210, target + tolerance)
    
    return lower, upper


def generate_target_sum_by_sine(draws: List[Dict]) -> int:
    """在正弦拟合预测范围内随机生成一个和值"""
    lower, upper = get_target_sum_sine_range(draws)
    return random.randint(lower, upper)

# ==================== 均值回归和值预测（容差±17）- 7码和值 ====================

def get_target_sum_mean_reversion(draws: List[Dict], num_count: int = 7, tolerance: int = 17) -> Tuple[int, int]:
    """
    均值回归和值预测（7码和值）
    
    返回:
        (目标值, 容差)
    """
    if len(draws) < 10:
        return 175, tolerance
    
    # 计算长期均值（最近100期7码和值）
    recent_draws = draws[-100:] if len(draws) >= 100 else draws
    all_sums = []
    for d in recent_draws:
        numbers = d.get('numbers', [])
        special = d.get('special', 0)
        sum_7 = sum(numbers) + special
        all_sums.append(sum_7)
    long_term_mean = np.mean(all_sums) if all_sums else 175
    long_term_std = np.std(all_sums) if len(all_sums) > 1 else 20
    
    # 短期均值（最近4期7码和值）
    short_draws = draws[-4:] if len(draws) >= 4 else draws
    short_sums = []
    for d in short_draws:
        numbers = d.get('numbers', [])
        special = d.get('special', 0)
        sum_7 = sum(numbers) + special
        short_sums.append(sum_7)
    short_mean = np.mean(short_sums) if short_sums else long_term_mean
    
    # 判断方向
    threshold = long_term_std * 0.15
    if short_mean > long_term_mean + threshold:
        target = int(long_term_mean - tolerance * 0.3)
    elif short_mean < long_term_mean - threshold:
        target = int(long_term_mean + tolerance * 0.3)
    else:
        target = int(long_term_mean)
    
    # 确保目标在合理范围内
    target = max(140, min(210, target))
    
    return target, tolerance


def get_target_sum_mean_reversion_range(draws: List[Dict], num_count: int = 7, tolerance: int = 17) -> Tuple[int, int]:
    """均值回归和值范围预测（7码和值）"""
    target, tol = get_target_sum_mean_reversion(draws, num_count, tolerance)
    lower = max(140, target - tol)
    upper = min(210, target + tol)
    return lower, upper


def generate_target_sum_by_mean_reversion(draws: List[Dict]) -> int:
    """在均值回归预测范围内随机生成一个和值"""
    lower, upper = get_target_sum_mean_reversion_range(draws)
    return random.randint(lower, upper)
#---------------------

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
# =====
# ==================== 方法A：分池评分法 - 核心评分函数 ====================

from math import comb
from collections import Counter
import numpy as np
import random

# ==================== 1. 辅助函数 ====================

def get_zone(num: int) -> int:
    """获取号码所在分区（1-7）"""
    return (num - 1) // 7 + 1


def get_zone_numbers(zone: int) -> list:
    """获取指定分区的所有号码"""
    start = (zone - 1) * 7 + 1
    end = start + 6
    return list(range(start, end + 1))


def calculate_absence(num: int, draws: list) -> int:
    """计算号码当前遗漏期数"""
    absence = 0
    for draw in reversed(draws):
        if num in draw['numbers']:
            break
        absence += 1
    return absence


# ==================== 2. 基础分（基于遗漏期数） ====================

def get_base_score(absence: int, hot_range: tuple = (0, 10)) -> int:
    hot_min, hot_max = hot_range
    if hot_min <= absence <= hot_max:
        return 55  # 热池基础分（从50提高到55）
    else:
        return 20  # 冷池基础分


# ==================== 3. 分区热度计算 ====================

def get_zone_rank(num: int, draws: list, window: int = 15) -> int:
    """
    获取号码所在分区的热度排名（1-7）
    window: 计算热区使用的期数，默认15期
    """
    zone = get_zone(num)
    
    # 计算最近window期各分区出现次数
    zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    for draw in draws[-window:]:
        for n in draw['numbers']:
            zone_counts[get_zone(n)] += 1
    
    # 排序获取排名
    sorted_zones = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)
    rank = [z for z, _ in sorted_zones].index(zone) + 1
    return rank

#------------
def get_zone_bonus(zone_rank: int, bonus_config: dict = None) -> int:
    """获取分区加分"""
    if bonus_config is None:
        bonus_config = {1: 12, 2: 8, 3: 5, 4: 0, 5: 0, 6: 0, 7: 0}
    return bonus_config.get(zone_rank, 0)


# ==================== 4. 规律加分 ====================

def is_gap_2(num: int, last_reds: list) -> bool:
    """夹号-间隔2：上期两码相差2，中间那个"""
    last_reds_sorted = sorted(last_reds)
    for i in range(len(last_reds_sorted) - 1):
        left = last_reds_sorted[i]
        right = last_reds_sorted[i + 1]
        if right - left == 2 and num == left + 1:
            return True
    return False


def is_gap_3(num: int, last_reds: list) -> bool:
    """夹号-间隔3：上期两码相差3，中间两个"""
    last_reds_sorted = sorted(last_reds)
    for i in range(len(last_reds_sorted) - 1):
        left = last_reds_sorted[i]
        right = last_reds_sorted[i + 1]
        if right - left == 3 and (num == left + 1 or num == left + 2):
            return True
    return False


def is_edge_to_normal(num: int, last_reds: list) -> bool:
    """边号-正码：与上期正码相差1"""
    for red in last_reds:
        if abs(num - red) == 1:
            return True
    return False


def is_edge_to_special(num: int, last_special: int) -> bool:
    """边号-特码：与上期特码相差1"""
    return last_special is not None and abs(num - last_special) == 1


def has_consecutive_potential(num: int, last_reds: list) -> bool:
    """连号潜力：上期连号向两端延伸"""
    last_reds_sorted = sorted(last_reds)
    for i in range(len(last_reds_sorted) - 1):
        if last_reds_sorted[i + 1] - last_reds_sorted[i] == 1:
            left_end = last_reds_sorted[i] - 1
            right_end = last_reds_sorted[i + 1] + 1
            if num == left_end or num == right_end:
                return True
    return False


def is_alternate_pattern(num: int, draws: list) -> bool:
    """隔期模式：与上上期号码重复"""
    if len(draws) < 2:
        return False
    last_last_draw = draws[-2]
    return num in last_last_draw['numbers']

#------------
def get_pattern_bonus(num: int, last_reds: list, last_special: int, draws: list,
                      pattern_config: dict = None) -> int:
    """计算规律加分总和（无上限，可叠加）"""
    if pattern_config is None:
        pattern_config = {
            "gap_2": 25,
            "gap_3": 12,
            "edge_normal": 15,
            "edge_special": 10,
            "consecutive": 8,
            "alternate": 8,
            "max": 999  # 无上限
        }
    
    bonus = 0
    
    # 夹号-间隔2（最强）
    if is_gap_2(num, last_reds):
        bonus += pattern_config["gap_2"]
    # 夹号-间隔3（次强）
    if is_gap_3(num, last_reds):  # 改为 if 而不是 elif，允许叠加
        bonus += pattern_config["gap_3"]
    
    # 正码边号
    if is_edge_to_normal(num, last_reds):
        bonus += pattern_config["edge_normal"]
    
    # 特码边号
    if is_edge_to_special(num, last_special):
        bonus += pattern_config["edge_special"]
    
    # 连号潜力
    if has_consecutive_potential(num, last_reds):
        bonus += pattern_config["consecutive"]
    
    # 隔期模式
    if is_alternate_pattern(num, draws):
        bonus += pattern_config["alternate"]
    
    return bonus  # 无上限


# ==================== 5. 冷码专有加分 ====================

def has_frequency_acceleration(num: int, draws: list, window: int = 10) -> bool:
    """
    频率加速度：最近5期出现次数 > 前5期的平均值
    """
    if len(draws) < window:
        return False
    
    # 最近5期出现次数
    recent_5 = 0
    for draw in draws[-5:]:
        if num in draw['numbers']:
            recent_5 += 1
    
    # 前5期出现次数
    prev_5 = 0
    for draw in draws[-10:-5]:
        if num in draw['numbers']:
            prev_5 += 1
    
    return recent_5 > prev_5


def has_consecutive_cold_appearance(num: int, draws: list) -> bool:
    """冷码连续2期出现"""
    if len(draws) < 2:
        return False
    return num in draws[-1]['numbers'] and num in draws[-2]['numbers']


def is_cold_return(num: int, draws: list, absence: int) -> bool:
    """冷号回补：遗漏超过20期后首次可能出现"""
    return absence >= 20 and num not in draws[-1]['numbers']


def has_cold_neighbor(num: int, draws: list) -> bool:
    """相邻冷号：与另一个冷号相邻（±1）"""
    last_draw = draws[-1]
    for n in last_draw['numbers']:
        if abs(num - n) == 1:
            return True
    return False


def get_cold_bonus(num: int, draws: list, absence: int,
                   cold_config: dict = None) -> int:
    """
    计算冷码专有加分（仅对冷池生效）
    cold_config: 自定义冷码加分配置，默认：
        {
            "frequency_acceleration": 12,
            "miss_13_15": 8,
            "consecutive": 10,
            "cold_return": 8,
            "cold_neighbor": 5,
            "max": 20
        }
    """
    # 只对冷码（遗漏>10期）生效
    if absence <= 10:
        return 0
    
    if cold_config is None:
        cold_config = {
            "frequency_acceleration": 12,
            "miss_13_15": 8,
            "consecutive": 10,
            "cold_return": 8,
            "cold_neighbor": 5,
            "max": 20
        }
    
    bonus = 0
    
    # 频率加速度
    if has_frequency_acceleration(num, draws):
        bonus += cold_config["frequency_acceleration"]
    
    # 遗漏13-15期
    if 13 <= absence <= 15:
        bonus += cold_config["miss_13_15"]
    
    # 连续2期出现
    if has_consecutive_cold_appearance(num, draws):
        bonus += cold_config["consecutive"]
    
    # 冷号回补
    if is_cold_return(num, draws, absence):
        bonus += cold_config["cold_return"]
    
    # 相邻冷号
    if has_cold_neighbor(num, draws):
        bonus += cold_config["cold_neighbor"]
    
    return min(bonus, cold_config["max"])


# ==================== 6. 综合评分函数 ====================

def calculate_method_a_score(num: int, draws: list, 
                              hot_range: tuple = (0, 10),
                              zone_window: int = 15,
                              zone_bonus_config: dict = None,
                              pattern_config: dict = None,
                              cold_config: dict = None) -> int:
    """
    方法A综合评分函数
    
    参数:
        num: 要计算的号码
        draws: 历史开奖数据
        hot_range: 热池遗漏范围，默认(0, 10)
        zone_window: 分区窗口期，默认15期
        zone_bonus_config: 分区加分配置
        pattern_config: 规律加分配置
        cold_config: 冷码专有加分配置
    
    返回:
        综合评分
    """
    # 1. 计算遗漏期数
    absence = calculate_absence(num, draws)
    
    # 2. 基础分
    base_score = get_base_score(absence, hot_range)
    
    # 3. 分区加分
    zone_rank = get_zone_rank(num, draws, zone_window)
    zone_bonus = get_zone_bonus(zone_rank, zone_bonus_config)
    
    # 4. 规律加分
    last_draw = draws[-1]
    last_reds = last_draw['numbers']
    last_special = last_draw.get('special')
    pattern_bonus = get_pattern_bonus(num, last_reds, last_special, draws, pattern_config)
    
    # 5. 冷码专有加分
    cold_bonus = get_cold_bonus(num, draws, absence, cold_config)
    
    # 6. 综合评分
    total_score = base_score + zone_bonus + pattern_bonus + cold_bonus
    # 冷池号码封顶49分
    if base_score == 20:  # 冷池基础分20
        total_score = min(total_score, 49)
    return total_score
# =====
# ==================== 方法A：分池评分法 - Softmax抽取与分池抽选 ====================

# ==================== 7. Softmax概率计算 ====================

def softmax_select(pool: List[int], scores: Dict[int, int], temperature: float = 0.8) -> int:
    """
    Softmax概率抽取单个号码
    
    参数:
        pool: 候选号码列表
        scores: 所有号码的评分字典
        temperature: 温度参数，默认0.8
    
    返回:
        被选中的号码
    """
    if not pool:
        return 0
    
    # 获取候选号码的分数列表
    score_list = [scores.get(num, 20) for num in pool]
    
    # 计算exp值
    exp_scores = np.exp(np.array(score_list) / temperature)
    
    # 计算概率
    probs = exp_scores / np.sum(exp_scores)
    
    # 按概率抽取
    selected = np.random.choice(pool, p=probs)
    return int(selected)

#--------------
def select_numbers_from_pool(pool: List[int], scores: Dict[int, int], 
                              count: int) -> List[int]:
    """
    从池中抽取指定数量的号码（不放回）- 线性归一化
    
    参数:
        pool: 候选号码列表
        scores: 所有号码的评分字典
        count: 需要抽取的号码个数
    
    返回:
        被选中的号码列表（已排序）
    """
    if count <= 0 or not pool:
        return []
    
    selected = []
    temp_pool = pool.copy()
    
    for _ in range(count):
        if not temp_pool:
            break
        
        # 获取当前池内所有号码的分数
        score_list = [scores.get(num, 0) for num in temp_pool]
        total_score = sum(score_list)
        
        if total_score <= 0:
            # 防错：如果总分为0，均匀抽取
            probs = [1 / len(temp_pool)] * len(temp_pool)
        else:
            # 线性归一化：概率 = 分数 / 总分
            probs = [s / total_score for s in score_list]
        
        # 按概率抽取
        selected_idx = np.random.choice(len(temp_pool), p=probs)
        selected_num = temp_pool[selected_idx]
        selected.append(selected_num)
        temp_pool.pop(selected_idx)
    
    return sorted(selected)


# ==================== 8. 池子划分 ====================

def split_pools_by_absence(draws: List[Dict], hot_range: tuple = (0, 10)) -> tuple:
    """
    根据遗漏期数划分热池和冷池
    
    参数:
        draws: 历史开奖数据
        hot_range: 热池遗漏范围，默认(0, 10)
    
    返回:
        (hot_pool, cold_pool) 热池和冷池的号码列表
    """
    hot_min, hot_max = hot_range
    hot_pool = []
    cold_pool = []
    
    for num in range(1, 50):
        absence = calculate_absence(num, draws)
        if hot_min <= absence <= hot_max:
            hot_pool.append(num)
        else:
            cold_pool.append(num)
    
    return hot_pool, cold_pool


# ==================== 9. 批量计算评分 ====================

def calculate_all_scores(draws: List[Dict], 
                         hot_range: tuple = (0, 10),
                         zone_window: int = 15,
                         zone_bonus_config: dict = None,
                         pattern_config: dict = None,
                         cold_config: dict = None) -> Dict[int, int]:
    """
    计算所有49个号码的综合评分
    
    返回:
        字典 {号码: 评分}
    """
    scores = {}
    for num in range(1, 50):
        score = calculate_method_a_score(
            num, draws, hot_range, zone_window, 
            zone_bonus_config, pattern_config, cold_config
        )
        scores[num] = score
    return scores


# ==================== 10. 和值目标获取 ====================

def get_sum_target_for_method_a(draws: List[Dict], num_count: int, 
                                 sum_predict_method: str = "动态回归") -> int:
    """
    获取方法A的和值目标（使用7码和值：正码+特码）
    """
    # 计算最近N期的7码和值（正码+特码）
    def get_7code_sums(draws, window):
        sums = []
        for draw in draws[-window:]:
            numbers = draw.get('numbers', [])
            special = draw.get('special', 0)
            sum_7 = sum(numbers) + special
            sums.append(sum_7)
        return sums
    
    # 动态回归
    if sum_predict_method == "动态回归":
        target, tolerance, _, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=4)
        offset = random.randint(-tolerance, tolerance)
        result = target + offset
        return max(140, min(210, result))
    
    elif sum_predict_method == "均值回归":
        # 均值回归：基于最近100期7码和值
        if len(draws) < 10:
            return random.randint(158, 192)
        
        all_sums = get_7code_sums(draws, 100)
        long_term_mean = np.mean(all_sums) if all_sums else 175
        long_term_std = np.std(all_sums) if len(all_sums) > 1 else 20
        
        short_sums = get_7code_sums(draws, 4)
        short_mean = np.mean(short_sums) if short_sums else long_term_mean
        
        threshold = long_term_std * 0.15
        if short_mean > long_term_mean + threshold:
            target = int(long_term_mean - 17 * 0.3)
        elif short_mean < long_term_mean - threshold:
            target = int(long_term_mean + 17 * 0.3)
        else:
            target = int(long_term_mean)
        
        target = max(140, min(210, target))
        lower = max(140, target - 17)
        upper = min(210, target + 17)
        
        return random.randint(lower, upper)
    
    elif sum_predict_method == "移动平均(7期)":
        # 移动平均：基于最近7期7码和值
        recent_sums = get_7code_sums(draws, 7)
        if len(recent_sums) < 7:
            return random.randint(158, 192)
        
        mean_val = np.mean(recent_sums)
        lower = max(140, int(mean_val) - 17)
        upper = min(210, int(mean_val) + 17)
        
        return random.randint(lower, upper)
    
    else:  # 正弦拟合
        # 正弦拟合：基于最近10期7码和值
        recent_sums = get_7code_sums(draws, 10)
        if len(recent_sums) < 10:
            return random.randint(158, 192)
        
        try:
            from scipy.optimize import curve_fit
            
            def sine_func(x, A, omega, phi, C):
                return A * np.sin(omega * x + phi) + C
            
            x = np.arange(10)
            y = np.array(recent_sums)
            A_guess = (np.max(y) - np.min(y)) / 2
            C_guess = np.mean(y)
            omega_guess = 2 * np.pi / 6.5
            
            params, _ = curve_fit(
                sine_func, x, y,
                p0=[A_guess, omega_guess, 0, C_guess],
                maxfev=2000
            )
            A, omega, phi, C = params
            next_val = sine_func(10, A, omega, phi, C)
            target = max(140, min(210, int(round(next_val))))
        except:
            target = int(round(np.mean(recent_sums)))
        
        lower = max(140, target - 17)
        upper = min(210, target + 17)
        
        return random.randint(lower, upper)


# ==================== 11. 和值筛选 ====================

def is_sum_valid(numbers: List[int], target_sum: int, tolerance: int = 17) -> bool:
    """检查组合和值是否在目标范围内"""
    total = sum(numbers)
    return abs(total - target_sum) <= tolerance


# ==================== 12. 方法A主函数：生成投注 ====================

def generate_bets_method_a(draws: List[Dict], num_bets: int, num_count: int = 7,
                           hot_count: int = 6, cold_count: int = 1,
                           hot_range: tuple = (0, 10),
                           hot_temperature: float = 0.8,
                           cold_temperature: float = 0.8,
                           zone_window: int = 15,
                           zone_bonus_config: dict = None,
                           pattern_config: dict = None,
                           cold_config: dict = None,
                           sum_predict_method: str = "移动平均(7期)",
                           random_seed: Optional[int] = None) -> List[Dict]:
    """
    方法A：分池评分法 - 生成投注
    
    参数:
        draws: 历史开奖数据
        num_bets: 生成组数
        num_count: 每注号码个数（固定7）
        hot_count: 热池抽取个数（默认6）
        cold_count: 冷池抽取个数（默认1）
        hot_range: 热池遗漏范围（默认0-10期）
        hot_temperature: 热池温度（默认0.8）
        cold_temperature: 冷池温度（默认0.8）
        zone_window: 分区窗口期（默认15期）
        zone_bonus_config: 分区加分配置
        pattern_config: 规律加分配置
        cold_config: 冷码加分配置
        sum_predict_method: 和值预测方法
        random_seed: 随机种子
    
    返回:
        投注列表
    """
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # 计算所有号码的评分
    all_scores = calculate_all_scores(
        draws, hot_range, zone_window, 
        zone_bonus_config, pattern_config, cold_config
    )
    
    # 划分池子
    hot_pool, cold_pool = split_pools_by_absence(draws, hot_range)
    
    bets = []
    base_target = get_target_sum_by_numbers_count(num_count)
    
    for i in range(num_bets):
        # 每注独立生成和值目标
        target_sum = get_sum_target_for_method_a(draws, num_count, sum_predict_method)
        tolerance = 17  # 默认容差
        
        # 尝试生成符合和值要求的组合
        max_attempts = 3000
        selected_numbers = None
        
        for attempt in range(max_attempts):
            # 从热池抽取（移除温度参数）
            hot_selected = select_numbers_from_pool(
                hot_pool, all_scores, hot_count
            )
            
            # 从冷池抽取（移除温度参数）
            cold_selected = select_numbers_from_pool(
                cold_pool, all_scores, cold_count
            )
            
            # 合并
            selected = sorted(hot_selected + cold_selected)
            
            # 验证和值
            if is_sum_valid(selected, target_sum, tolerance):
                selected_numbers = selected
                break
        
        # 如果没找到符合和值的，用最后一次生成的
        if selected_numbers is None:
            hot_selected = select_numbers_from_pool(
                hot_pool, all_scores, hot_count, hot_temperature
            )
            cold_selected = select_numbers_from_pool(
                cold_pool, all_scores, cold_count, cold_temperature
            )
            selected_numbers = sorted(hot_selected + cold_selected)
        
        total = sum(selected_numbers)
        
        bets.append({
            'numbers': selected_numbers,
            'sum': total,
            'target': f'方法A(目标{target_sum})',
            'deviation': total - target_sum
        })
                    
    return bets


# ==================== 13. 获取方法A的详细评分数据（用于选页展示） ====================

def get_method_a_score_details(draws: List[Dict],
                               hot_range: tuple = (0, 10),
                               zone_window: int = 15,
                               zone_bonus_config: dict = None,
                               pattern_config: dict = None,
                               cold_config: dict = None,
                               hot_temperature: float = None,
                               cold_temperature: float = None) -> Dict:
    """
    获取方法A的详细评分数据，用于选页展示
    
    返回:
        {
            'hot_pool': List[int],           # 热池号码列表
            'cold_pool': List[int],          # 冷池号码列表
            'scores': Dict[int, int],        # 所有号码评分
            'hot_probs': Dict[int, float],   # 热池内抽出概率
            'cold_probs': Dict[int, float],  # 冷池内抽出概率
            'zone_rank': Dict[int, int],     # 各分区排名
            'zone_counts': Dict[int, int],   # 各分区出现次数
        }
    """
    # 计算所有号码评分
    scores = calculate_all_scores(
        draws, hot_range, zone_window, 
        zone_bonus_config, pattern_config, cold_config
    )
    
    # 划分池子
    hot_pool, cold_pool = split_pools_by_absence(draws, hot_range)
    
    # ========== 计算热池内抽出概率（线性归一化） ==========
    hot_probs_dict = {}
    if hot_pool:
        hot_scores = [scores[num] for num in hot_pool]
        total_hot_score = sum(hot_scores)
        if total_hot_score > 0:
            hot_probs = [s / total_hot_score for s in hot_scores]
            hot_probs_dict = {num: prob for num, prob in zip(hot_pool, hot_probs)}
        else:
            prob = 1 / len(hot_pool)
            hot_probs_dict = {num: prob for num in hot_pool}
    
    # ========== 计算冷池内抽出概率（线性归一化） ==========
    cold_probs_dict = {}
    if cold_pool:
        cold_scores = [scores[num] for num in cold_pool]
        total_cold_score = sum(cold_scores)
        if total_cold_score > 0:
            cold_probs = [s / total_cold_score for s in cold_scores]
            cold_probs_dict = {num: prob for num, prob in zip(cold_pool, cold_probs)}
        else:
            prob = 1 / len(cold_pool)
            cold_probs_dict = {num: prob for num in cold_pool}
    
    # 计算分区排名
    zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    for draw in draws[-zone_window:]:
        for num in draw['numbers']:
            zone_counts[get_zone(num)] += 1
    
    sorted_zones = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)
    zone_rank = {zone: rank + 1 for rank, (zone, _) in enumerate(sorted_zones)}
    
    return {
        'hot_pool': hot_pool,
        'cold_pool': cold_pool,
        'scores': scores,
        'hot_probs': hot_probs_dict,
        'cold_probs': cold_probs_dict,
        'zone_rank': zone_rank,
        'zone_counts': zone_counts,
    }
# =====

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
#   4. 修复parse_excel_file缩进问题
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
#---------------
def fetch_latest_from_9800() -> List[Dict]:
    """
    从 http://www.9800.com.tw/lotto6/statistics.html 抓取最近20期数据
    """
    url = "http://www.9800.com.tw/lotto6/statistics.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = 'utf-8'
        if resp.status_code != 200:
            st.error(f"网页返回异常状态码: {resp.status_code}")
            return []
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        st.error(f"网络请求失败: {e}")
        return []

    # 定位包含数据的主表格
    tables = soup.find_all('table')
    target_table = None
    for tbl in tables:
        text = tbl.get_text()
        if '期次' in text and '開獎日期' in text:
            rows = tbl.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 1 and cells[0].get_text(strip=True).isdigit():
                    target_table = tbl
                    break
            if target_table:
                break

    if not target_table:
        st.error("未找到数据表格，网页结构可能已变化")
        return []

    rows = target_table.find_all('tr')
    data = []
    for idx, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) < 9:
            continue
        # 跳过表头
        header_text = cells[0].get_text(strip=True)
        if header_text == '期次':
            continue
        try:
            period_str = cells[0].get_text(strip=True)
            if not period_str.isdigit():
                continue
            period = int(period_str)
            date_str = cells[1].get_text(strip=True)
            numbers = []
            for i in range(2, 8):
                num = int(cells[i].get_text(strip=True))
                numbers.append(num)
            special = int(cells[8].get_text(strip=True))
            data.append({
                'period': period,
                'date': date_str,
                'numbers': sorted(numbers),
                'special': special,
                'sum': sum(numbers) + special
            })
        except (ValueError, IndexError):
            continue

    return data
#------------------
def parse_excel_file(uploaded_file) -> Optional[List[Dict]]:
    """
    解析Excel文件 - 完整修复版
    支持：精确列名匹配、位置匹配、日期格式自动识别
    """
    try:
        # 读取Excel，强制第一行为表头
        df = pd.read_excel(uploaded_file, sheet_name=0, header=0)
        
        # 打印列名（用于调试，在终端可见）
        print(f"Excel列名: {df.columns.tolist()}")
        
        # ========== 1. 精确匹配列名 ==========
        period_col = None
        date_col = None
        number_cols = []
        special_col = None
        
        for col in df.columns:
            col_str = str(col).strip()
            
            if col_str == '期次':
                period_col = col
            elif col_str == '開獎日期':
                date_col = col
            elif col_str in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
                number_cols.append(col)
            elif col_str == 'B7':
                special_col = col
        
        # ========== 2. 如果精确匹配失败，使用位置匹配 ==========
        if period_col is None and len(df.columns) > 0:
            period_col = df.columns[0]
            st.info(f"使用位置匹配: 期次列 = {period_col}")
        
        if date_col is None and len(df.columns) > 1:
            date_col = df.columns[1]
            st.info(f"使用位置匹配: 開獎日期列 = {date_col}")
        
        if len(number_cols) != 6 and len(df.columns) >= 8:
            number_cols = df.columns[2:8].tolist()
            st.info(f"使用位置匹配: 正码列 = {[str(c) for c in number_cols]}")
        
        if special_col is None and len(df.columns) > 8:
            special_col = df.columns[8]
            st.info(f"使用位置匹配: 特码列 = {special_col}")
        
        # ========== 3. 验证 ==========
        if period_col is None:
            st.error("无法识别期次列")
            return None
        if date_col is None:
            st.error("无法识别開獎日期列")
            return None
        if len(number_cols) != 6:
            st.error(f"无法识别正码列，需要6列，找到{len(number_cols)}列")
            return None
        if special_col is None:
            st.error("无法识别特码列")
            return None
        
        # ========== 4. 辅助函数：日期转换 ==========
        def convert_to_date_string(value):
            if pd.isna(value):
                return None
            if isinstance(value, datetime):
                return value.strftime('%Y-%m-%d')
            if isinstance(value, str):
                date_part = value.split()[0] if ' ' in value else value
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']:
                    try:
                        dt = datetime.strptime(date_part, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                return date_part[:10]
            if isinstance(value, (int, float)):
                try:
                    base_date = datetime(1900, 1, 1)
                    delta_days = int(value) - 2
                    dt = base_date + timedelta(days=delta_days)
                    return dt.strftime('%Y-%m-%d')
                except:
                    return None
            return str(value)[:10]
        
        # ========== 5. 解析数据 ==========
        draws = []
        error_count = 0
        skip_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 期次
                period_val = row[period_col]
                if pd.isna(period_val):
                    skip_count += 1
                    continue
                
                if isinstance(period_val, (int, float)):
                    period = int(period_val)
                else:
                    period_str = str(period_val).strip()
                    if period_str.isdigit():
                        period = int(period_str)
                    else:
                        period = period_str
                
                # 日期
                date_val = row[date_col]
                date_str = convert_to_date_string(date_val)
                
                # 正码
                nums = []
                for col in number_cols:
                    val = row[col]
                    if pd.isna(val):
                        raise ValueError(f"正码为空")
                    num = int(float(val))
                    if not (1 <= num <= 49):
                        raise ValueError(f"正码 {num} 超出范围")
                    nums.append(num)
                
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
        
        if skip_count > 0:
            st.info(f"跳过 {skip_count} 行（期次为空）")
        if error_count > 0:
            st.warning(f"跳过 {error_count} 行（数据格式错误）")
        
        if not draws:
            st.error("未找到有效数据")
            return None
        
        draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
        
        first_period = draws[0].get('period')
        last_period = draws[-1].get('period')
        st.success(f"✅ 成功解析 {len(draws)} 期数据 (范围: {first_period} - {last_period})")
        
        return draws
        
    except Exception as e:
        st.error(f"Excel解析错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ==================== 回测核心函数 ====================
def calculate_7code_prize(bet_numbers: List[int], draw: Dict) -> int:
    from math import comb
    
    draw_numbers = set(draw['numbers'])
    draw_special = draw.get('special')
    
    N = len(bet_numbers)
    A = len(set(bet_numbers) & draw_numbers)
    has_special = draw_special is not None and draw_special in bet_numbers
    
    PRIZES = {
        7: 20, 6: 160, 5: 320, 4: 4800, 3: 30800, 2: 1530000, 1: 5090000,
    }
    
    non_win_count = N - A
    if has_special:
        non_win_count -= 1
    non_win_count = max(0, non_win_count)
    
    # 详细日志
    print(f"\n[奖金计算] N={N}, A={A}, has_special={has_special}, non_win_count={non_win_count}")
    
    total_prize = 0
    
    # 第7组：中3码（无特码）
    if A >= 3 and non_win_count >= 3:
        count = comb(A, 3) * comb(non_win_count, 3)
        prize = count * PRIZES[7]
        print(f"  - 中3码(无特): count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第6组：中3码+特码
    if has_special and A >= 3 and non_win_count >= 2:
        count = comb(A, 3) * comb(non_win_count, 2)
        prize = count * PRIZES[6]
        print(f"  - 中3码+特: count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第5组：中4码（无特码）
    if A >= 4 and non_win_count >= 2:
        count = comb(A, 4) * comb(non_win_count, 2)
        prize = count * PRIZES[5]
        print(f"  - 中4码(无特): count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第4组：中4码+特码
    if has_special and A >= 4 and non_win_count >= 1:
        count = comb(A, 4) * comb(non_win_count, 1)
        prize = count * PRIZES[4]
        print(f"  - 中4码+特: count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第3组：中5码（无特码）
    if A >= 5 and non_win_count >= 1:
        count = comb(A, 5) * comb(non_win_count, 1)
        prize = count * PRIZES[3]
        print(f"  - 中5码(无特): count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第2组：中5码+特码
    if has_special and A >= 5:
        count = comb(A, 5) * comb(non_win_count, 0)
        prize = count * PRIZES[2]
        print(f"  - 中5码+特: count={count}, 奖金=${prize}")
        total_prize += prize
    
    # 第1组：中6码
    if A >= 6:
        count = comb(A, 6) * comb(non_win_count, 0)
        prize = count * PRIZES[1]
        print(f"  - 中6码: count={count}, 奖金=${prize}")
        total_prize += prize
    
    print(f"  => 总奖金 = ${total_prize}")
    
    return total_prize
#----------------
def get_best_match_score(bet_numbers: List[int], draw: Dict) -> float:
    """
    获取N码复式中最好的匹配分数（用于日志显示）
    
    返回:
        最佳匹配分数，如 3.0（中3码）, 3.5（中3+特）, 4.0（中4码）等
    """
    from itertools import combinations
    
    draw_numbers = set(draw['numbers'])
    draw_special = draw.get('special')
    
    best_score = 0
    
    # 枚举所有C(N,6)种6码组合
    for combo in combinations(bet_numbers, 6):
        match_count = len(set(combo) & draw_numbers)
        has_special_in_combo = draw_special is not None and draw_special in combo
        
        score = match_count
        if has_special_in_combo and match_count >= 3:
            score += 0.5
        
        best_score = max(best_score, score)
    
    return best_score
#----------------
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


# ==================== 管理员页面 ====================
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
            reverse=True
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
                '和值': d.get('sum_7', d.get('sum', 0))
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
    #---------
    # 可编辑表格
    with st.form(key="data_editor_form"):
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            key="marksix_data_editor",  # 修改 key 避免与双色球冲突
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
        with col_save2:
            incremental_submitted = st.form_submit_button("🔄 增量同步保存", use_container_width=True)
        
        # ========== 全量覆盖保存 ==========
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
                        new_draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
                        if save_draws_to_supabase(new_draws):
                            st.session_state['draws_loaded'] = new_draws
                            st.success(f"全量覆盖保存 {len(new_draws)} 期数据成功！")
                            st.rerun()
                        else:
                            st.error("保存失败")
                    else:
                        st.error("没有有效数据可保存")
        
        # ========== 增量同步保存 ==========
        if incremental_submitted:
            if edited_df is None or len(edited_df) == 0:
                st.error("没有数据可同步")
            else:
                with st.spinner("正在执行增量同步..."):
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
                        new_draws.sort(key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
                        result = incremental_sync_draws(new_draws)
                        st.success(f"增量同步完成：新增 {result['inserted']} 期，更新 {result['updated']} 期，删除 {result['deleted']} 期")
                        # 重新加载数据到session
                        refreshed_draws = load_draws_from_supabase()
                        if refreshed_draws:
                            st.session_state['draws_loaded'] = refreshed_draws
                        st.rerun()
                    else:
                        st.error("没有有效数据可同步")
    
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
                # 预览最新10期
                preview_data = []
                for d in excel_draws[-10:]:
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

print("第2部分加载完成 (v7.1 - 修复版)")
print("=" * 60)
print("请确认第2部分代码，输入 CONFIRM 后继续第3部分")
print("=" * 60)
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
    max_attempts = 3000
    for _ in range(max_attempts):
        selected = weighted_random_sample(weights, k=num_count)
        if is_valid_combination(selected, target_sum, tolerance):
            return selected, sum(selected)
    
    for _ in range(1000):
        selected = weighted_random_sample(weights, k=num_count)
        total = sum(selected)
        if abs(total - target_sum) <= tolerance + 5:
            return selected, total
    
    return sorted(random.sample(range(1, 50), num_count)), sum(sorted(random.sample(range(1, 50), num_count)))


# ==================== 方法1：当前方法 ====================
def generate_bets_method1_current(draws: List[Dict], num_bets: int, num_count: int,
                                   trend_window: int, random_seed: Optional[int],
                                   analysis_periods: int, sum_predict_method: str) -> List[Dict]:

    """
    方法1：当前方法
    - 冷热码分析
    - 用户选择的预测方法（动态回归/均值回归/移动平均/正弦拟合）
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
    
    base_target = get_target_sum_by_numbers_count(num_count)
    
    bets = []
    for i in range(num_bets):
        # 使用用户选择的预测方法
        t = get_sum_target_by_method(draws, num_count, trend_window, sum_predict_method)
        tolerance = 17  # 默认容差，动态回归的容差已在函数内部处理
        t = max(140, min(210, t))  # 7码范围限制
        
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

#----------------
def generate_bets_method2_hybrid(draws: List[Dict], num_bets: int, num_count: int,
                                   trend_window: int, random_seed: Optional[int],
                                   analysis_periods: int, sum_predict_method: str) -> List[Dict]:
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
    #---------------
    # 和值预测（使用用户选择的方法）
    base_target = get_target_sum_by_numbers_count(num_count)
    
    # 选择胆码
    anchors = select_anchor_numbers(draws, num_anchors=3, analysis_periods=analysis_periods)
    
    # 调整胆码权重（降低避免重复）
    for a in anchors:
        if a in weights:
            weights[a] *= 0.3
    
    bets = []
    for i in range(num_bets):
        # 每注独立生成和值目标
        t = get_sum_target_by_method(draws, num_count, trend_window, sum_predict_method)
        
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

#-------------
def train_lightgbm_model(draws: List[Dict], lookback: int = 100, random_seed: int = 7) -> Optional[Any]:
    """训练LightGBM模型"""
    if not LGB_AVAILABLE:
        return None
    
    X, y = prepare_lightgbm_dataset(draws, lookback=lookback)
    if X is None or len(X) < 100:
        return None
    
    try:
        #-----
        if lookback <= 20:
            # 极简模式：深度只有2，叶子只有4个，强L2正则化
            model = lgb.LGBMClassifier(
                n_estimators=30,          # 树极少，防止过拟合
                max_depth=2,              # 深度2，只能捕捉最简单的线性/交互
                num_leaves=4,             # 叶子数极少
                reg_lambda=100,           # 极强L2正则化（惩罚大权重）
                reg_alpha=10,             # 极强L1正则化（特征选择）
                learning_rate=0.05,       # 降低学习率
                random_state=random_seed,
                verbose=-1,
                min_child_samples=3       # 叶子最少3个样本
            )
        else:
            # 原来的正常模式（大于20期时使用）
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_seed,
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

#----
def generate_bets_method3_lightgbm(draws: List[Dict], num_bets: int, num_count: int,
                                     trend_window: int, random_seed: Optional[int],
                                     lightgbm_lookback: int = 100, sum_predict_method: str = "移动平均(7期)") -> List[Dict]:
    """
    方法3：LightGBM - 添加和值筛选（500次 + 降级策略）
    """
    if not LGB_AVAILABLE:
        return generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, 50, sum_predict_method)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    model = train_lightgbm_model(draws, lookback=lightgbm_lookback)
    
    if model is None:
        return generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, 50, sum_predict_method)
    
    predicted_numbers = predict_with_lightgbm(model, draws)
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    base_target = get_target_sum_by_numbers_count(num_count)
    max_attempts = 500
    
    bets = []
    for i in range(num_bets):
        # 每注独立生成和值目标
        target_sum = get_sum_target_by_method(draws, num_count, trend_window, sum_predict_method)
        
        selected_numbers = None
        
        # 第1轮：正常容差（±17）
        for attempt in range(max_attempts):
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
            
            if abs(total - target_sum) <= 17:
                selected_numbers = nums
                break
        
        # 第2轮：放宽容差到 ±25
        if selected_numbers is None:
            for attempt in range(max_attempts):
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
                
                if abs(total - target_sum) <= 25:
                    selected_numbers = nums
                    break
        
        # 第3轮：保底（无和值筛选）
        if selected_numbers is None:
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
            
            selected_numbers = sorted(nums[:num_count])
        
        total = sum(selected_numbers)
        
        bets.append({
            'numbers': selected_numbers,
            'sum': total,
            'target': f'LightGBM(目标{target_sum})',
            'deviation': total - base_target
        })
    
    return bets


# ==================== 方法4：XGBoost + 神经网络集成 ====================
def build_advanced_features(draws: List[Dict], target_num: int) -> Optional[Dict]:
    """构建高级特征（包含更多规律特征）"""
    if len(draws) < 15:
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

#-----------
def prepare_advanced_dataset(draws: List[Dict], lookback: int = 200) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """准备高级数据集"""
    # 放宽检查，与 build_advanced_features 保持一致
    if len(draws) < 15:
        return None, None
    # 后续循环中，i 从 lookback 开始，如果 lookback > len(draws)，会直接返回空
    if lookback >= len(draws):
        return None, None
    # ...
    
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

#--------------------
def train_xgboost_nn_ensemble(draws: List[Dict], lookback: int = 100, random_seed: int = 7) -> Optional[Dict]:
    """训练XGBoost + 神经网络集成（优化版）"""
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    
    X, y = prepare_advanced_dataset(draws, lookback=lookback)
    if X is None or len(X) < 100:
        return None
    
    try:
        # ========== 策略 A：极简特征 + 极端正则化 ==========
        if lookback <= 20:
            xgb_model = xgb.XGBClassifier(
                n_estimators=30,          # 树极少
                max_depth=2,              # 深度只有2
                learning_rate=0.05,
                reg_lambda=100,           # 极强L2
                reg_alpha=10,             # 极强L1
                gamma=5,                  # 分裂所需的最小损失减少（越大越保守）
                min_child_weight=3,       # 叶子最小权重
                random_state=random_seed,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            # 原来的正常模式（大于20期时使用）
            xgb_model = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_seed,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        # ===================================================
        
        nn_model = MLPClassifier(
            hidden_layer_sizes=(32, 16),  # 从3层减少到2层
            activation='relu',
            max_iter=100,                  # 从200降到100
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

#------------------
def generate_bets_method4_ensemble(draws: List[Dict], num_bets: int, num_count: int,
                                     trend_window: int, random_seed: Optional[int],
                                     ensemble_lookback: int = 100, sum_predict_method: str = "移动平均(7期)") -> List[Dict]:
    """
    方法4：XGBoost + 神经网络集成 - 添加和值筛选（500次 + 降级策略）
    """
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, 100, sum_predict_method)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    ensemble = train_xgboost_nn_ensemble(draws, lookback=ensemble_lookback)
    
    if ensemble is None:
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, 100, sum_predict_method)
    
    predicted_numbers = predict_with_ensemble(ensemble, draws)
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    base_target = get_target_sum_by_numbers_count(num_count)
    max_attempts = 500
    
    bets = []
    for i in range(num_bets):
        # 每注独立生成和值目标
        target_sum = get_sum_target_by_method(draws, num_count, trend_window, sum_predict_method)
        
        selected_numbers = None
        
        # 第1轮：正常容差（±17）
        for attempt in range(max_attempts):
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
            
            if abs(total - target_sum) <= 17:
                selected_numbers = nums
                break
        
        # 第2轮：放宽容差到 ±25
        if selected_numbers is None:
            for attempt in range(max_attempts):
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
                
                if abs(total - target_sum) <= 25:
                    selected_numbers = nums
                    break
        
        # 第3轮：保底（无和值筛选）
        if selected_numbers is None:
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
            
            selected_numbers = sorted(nums[:num_count])
        
        total = sum(selected_numbers)
        
        bets.append({
            'numbers': selected_numbers,
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
    """
    方法5：综合模式（运行方法1-4，取高频号码 + 规律加权）- 添加和值筛选
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
    
    all_numbers = []
    
    # 方法1
    bets1 = generate_bets_method1_current(draws, num_bets, num_count, trend_window, random_seed, method1_window, "移动平均(7期)")
    for bet in bets1:
        all_numbers.extend(bet['numbers'])
    
    # 方法2
    bets2 = generate_bets_method2_hybrid(draws, num_bets, num_count, trend_window, random_seed, method2_window, "移动平均(7期)")
    for bet in bets2:
        all_numbers.extend(bet['numbers'])
    
    # 方法3
    bets3 = generate_bets_method3_lightgbm(draws, num_bets, num_count, trend_window, random_seed, method3_window, "移动平均(7期)")
    for bet in bets3:
        all_numbers.extend(bet['numbers'])
    
    # 方法4
    bets4 = generate_bets_method4_ensemble(draws, num_bets, num_count, trend_window, random_seed, method4_window, "移动平均(7期)")
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
    
    base_target = get_target_sum_by_numbers_count(num_count)
    max_attempts = 500
    
    bets = []
    for i in range(num_bets):
        # 每注独立生成和值目标
        target_sum = get_sum_target_by_method(draws, num_count, trend_window, "移动平均(7期)")
        
        selected_numbers = None
        
        # 第1轮：正常容差（±17）
        for attempt in range(max_attempts):
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
            if abs(total - target_sum) <= 17:
                selected_numbers = nums
                break
        
        # 第2轮：放宽容差到 ±25
        if selected_numbers is None:
            for attempt in range(max_attempts):
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
                if abs(total - target_sum) <= 25:
                    selected_numbers = nums
                    break
        
        # 第3轮：保底（无和值筛选）
        if selected_numbers is None:
            nums = final_numbers.copy()
            if i > 0:
                replace_count = min(i, 2)
                for _ in range(replace_count):
                    idx = random.randint(0, len(nums) - 1)
                    candidates = [n for n in range(1, 50) if n not in nums]
                    if candidates:
                        nums[idx] = random.choice(candidates)
                nums = sorted(nums)
            selected_numbers = nums
        
        total = sum(selected_numbers)
        
        bets.append({
            'numbers': selected_numbers,
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
#---------------------
# ==================== 统一的和值目标获取函数 ====================

def get_sum_target_by_method(draws: List[Dict], num_count: int, trend_window: int, sum_predict_method: str) -> int:
    """
    根据用户选择的预测方法返回和值目标（每注独立随机）
    """
    if sum_predict_method == "动态回归":
        target, tolerance, _, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
        offset = random.randint(-tolerance, tolerance)
        result = target + offset
        # 限制在7码合理范围内
        return max(140, min(210, result))
    elif sum_predict_method == "均值回归":
        lower, upper = get_target_sum_mean_reversion_range(draws, num_count)
    elif sum_predict_method == "移动平均(7期)":
        lower, upper = get_target_sum_moving_average_range(draws)
    else:  # 正弦拟合
        lower, upper = get_target_sum_sine_range(draws)
    
    return random.randint(lower, upper)

#-------------------------
def run_backtest_single_method(draws: List[Dict], method_key: str, num_bets: int, num_count: int,
                                 trend_window: int, test_periods: int, train_window: int,
                                 seed_mode: str, fixed_seed_value: int,
                                 sum_predict_method: str = "移动平均(7期)") -> Optional[Dict]:
    """
    单方法回测 - 完整修复版
    
    修复内容：
    1. 使用正确的N码复式奖金计算（支持6-10码）
    2. 使用正确的匹配分数计算
    3. 成本计算：每组成本 = C(num_count, 6) × 5（半注）
    4. 修复缓存key包含num_count，避免不同号码数的缓存冲突
    """
    if len(draws) < train_window + test_periods:
        return None
    
    method_seed_offset = {"方法1": 100, "方法2": 200, "方法3": 300, "方法4": 400, "方法5": 500}.get(method_key, 0)
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    prize_details = []
    
    # 缓存投注结果（每10期重新训练一次）
    trained_models = {}
    retrain_interval = 10
    
    # 计算每组成本（半注）
    from math import comb
    cost_per_bet = comb(num_count, 6) * 5  # C(N,6) × $5
    
    for idx in range(test_periods):
        i = idx
        # 训练数据：使用当期之前的数据
        train_draws = draws[:-(test_periods - i)]
        test_draw = draws[-(test_periods - i)]
        test_period = test_draw.get('period', '')
        
        # 设置随机种子
        if seed_mode == "date":
            test_date = test_draw.get('date')
            if test_date:
                try:
                    dt = datetime.strptime(test_date[:10], '%Y-%m-%d')
                    seed_val = int(datetime(dt.year, dt.month, dt.day, 21, 30).timestamp())
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
        
        # 每 retrain_interval 期重新训练一次
        # 修复：缓存key中加入 num_count，避免不同号码数的缓存冲突
        model_key = f"{method_key}_{num_count}_{i // retrain_interval}"
        
        if model_key not in trained_models:
            if method_key == "方法1":
                bets = generate_bets_method1_current(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
            elif method_key == "方法2":
                bets = generate_bets_method2_hybrid(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
            elif method_key == "方法3":
                bets = generate_bets_method3_lightgbm(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
            elif method_key == "方法4":
                bets = generate_bets_method4_ensemble(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
            else:  # 方法5: 综合模式
                bets1 = generate_bets_method1_current(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
                bets2 = generate_bets_method2_hybrid(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
                bets3 = generate_bets_method3_lightgbm(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
                bets4 = generate_bets_method4_ensemble(train_draws, num_bets, num_count, trend_window, seed_val, train_window, sum_predict_method)
                
                # 合并所有投注，统计频率
                all_numbers = []
                for bet in bets1 + bets2 + bets3 + bets4:
                    all_numbers.extend(bet['numbers'])
                
                freq_counter = Counter(all_numbers)
                top_numbers = [num for num, _ in freq_counter.most_common(num_count)]
                
                # 生成多组投注
                bets = []
                for j in range(num_bets):
                    nums = top_numbers.copy()
                    if j > 0:
                        replace_count = min(j, 2)
                        for _ in range(replace_count):
                            idx2 = random.randint(0, len(nums) - 1)
                            candidates = [n for n in range(1, 50) if n not in nums]
                            if candidates:
                                nums[idx2] = random.choice(candidates)
                        nums = sorted(nums)
                    bets.append({'numbers': nums, 'sum': sum(nums)})
            
            trained_models[model_key] = bets
        else:
            bets = trained_models[model_key]
        
        # 计算最佳匹配的奖金和匹配分数
        best_prize = 0
        best_match_score = 0
        
        for bet in bets:
            # 使用修复后的奖金计算函数
            prize = calculate_7code_prize(bet['numbers'], test_draw)
            
            # 使用修复后的匹配分数计算
            match_score = get_best_match_score(bet['numbers'], test_draw)
            
            if prize > best_prize:
                best_prize = prize
                best_match_score = match_score
        
        # 成本计算（半注）
        total_cost += num_bets * cost_per_bet
        total_prize += best_prize
        
        if best_prize > 0:
            win_count += 1
            # 格式化奖金显示（按金额从大到小排列）
            if best_prize >= 5090000:
                prize_desc = "509万"
            elif best_prize >= 1530000:
                prize_desc = "153万"
            elif best_prize >= 30800:
                prize_desc = "3.08万"
            elif best_prize >= 10560:
                prize_desc = "10,560"
            elif best_prize >= 1040:
                prize_desc = "1,040"
            elif best_prize >= 4800:
                prize_desc = "4,800"
            elif best_prize >= 1600:
                prize_desc = "1,600"
            elif best_prize >= 520:
                prize_desc = "520"
            elif best_prize >= 320:
                prize_desc = "320"
            elif best_prize >= 160:
                prize_desc = "160"
            elif best_prize >= 80:
                prize_desc = "80"
            elif best_prize >= 40:
                prize_desc = "40"
            else:
                prize_desc = str(best_prize)
            
            # 格式化匹配分数显示
            if best_match_score == int(best_match_score):
                match_display = f"{int(best_match_score)}"
            else:
                match_display = f"{best_match_score:.1f}"
            
            prize_details.append(f"{test_period}({match_display}, {prize_desc})")
    
    net = total_prize - total_cost
    roi = (net / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (win_count / test_periods) * 100 if test_periods > 0 else 0
    
    name_map = {
        "方法1": "方法1: 当前方法",
        "方法2": "方法2: 胆拖混合",
        "方法3": "方法3: LightGBM",
        "方法4": "方法4: XGBoost+NN",
        "方法5": "方法5: 综合模式"
    }
    
    return {
        "方法": name_map.get(method_key, method_key),
        "ROI": roi,
        "总成本": total_cost,
        "总奖金": total_prize,
        "净收益": net,
        "中奖率": win_rate,
        "中奖明细": ", ".join(prize_details) if prize_details else "无"
    }
#---------------
# ==================== 方法B：新胆拖混合（基于方法A评分） ====================

def generate_bets_method_b(draws: List[Dict], num_bets: int, num_count: int = 7,
                           sum_predict_method: str = "移动平均(7期)",
                           random_seed: Optional[int] = None) -> List[Dict]:
    """
    方法B：新胆拖混合（基于方法A评分）
    
    逻辑：
    1. 计算方法A评分
    2. 从热池评分Top10中随机抽取3个作为胆码
    3. 从剩余热池（排除Top10）中按概率抽取3个作为热池拖码
    4. 从冷池中按概率抽取1个作为冷池拖码
    5. 合并胆码+热池拖码+冷池拖码，和值筛选
    
    参数:
        draws: 历史开奖数据
        num_bets: 生成组数
        num_count: 每注号码个数（固定7）
        sum_predict_method: 和值预测方法
        random_seed: 随机种子
    
    返回:
        投注列表
    """
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # 获取方法A的配置（从session_state）
    zone_bonus_config = {
        1: st.session_state.get('zone1_bonus', 12),
        2: st.session_state.get('zone2_bonus', 8),
        3: st.session_state.get('zone3_bonus', 5),
        4: st.session_state.get('zone4_bonus', 0),
        5: st.session_state.get('zone5_bonus', 0),
        6: st.session_state.get('zone6_bonus', 0),
        7: st.session_state.get('zone7_bonus', 0),
    }
    
    pattern_config = {
        "gap_2": st.session_state.get('pattern_gap2', 25),
        "gap_3": st.session_state.get('pattern_gap3', 12),
        "edge_normal": st.session_state.get('pattern_edge_normal', 15),
        "edge_special": st.session_state.get('pattern_edge_special', 10),
        "consecutive": st.session_state.get('pattern_consecutive', 8),
        "alternate": st.session_state.get('pattern_alternate', 8),
        "max": 999  # 无上限
    }
    
    cold_config = {
        "frequency_acceleration": st.session_state.get('cold_freq_acc', 12),
        "miss_13_15": st.session_state.get('cold_miss_13_15', 8),
        "consecutive": st.session_state.get('cold_consecutive', 10),
        "cold_return": st.session_state.get('cold_return', 8),
        "cold_neighbor": st.session_state.get('cold_neighbor', 5),
        "max": st.session_state.get('cold_max', 20),
    }
    
    # 热池遗漏范围
    hot_range = (st.session_state.get('method_a_hot_range_start', 0),
                 st.session_state.get('method_a_hot_range_end', 10))
    zone_window = st.session_state.get('method_a_zone_window', 15)
    
    # 计算所有号码的评分
    all_scores = calculate_all_scores(
        draws, hot_range, zone_window,
        zone_bonus_config, pattern_config, cold_config
    )
    
    # 划分池子
    hot_pool, cold_pool = split_pools_by_absence(draws, hot_range)
    
    # 获取热池Top10（评分最高的10个号码）
    hot_scores = [(num, all_scores[num]) for num in hot_pool]
    hot_scores.sort(key=lambda x: x[1], reverse=True)
    top10_hot = [num for num, _ in hot_scores[:10]]
    
    # 剩余热池（排除Top10）
    remaining_hot_pool = [num for num in hot_pool if num not in top10_hot]
    
    bets = []
    base_target = get_target_sum_by_numbers_count(num_count)
    
    for i in range(num_bets):
        # 每注独立生成和值目标
        target_sum = get_sum_target_for_method_a(draws, num_count, sum_predict_method)
        tolerance = 17
        
        # 尝试生成符合和值要求的组合
        max_attempts = 3000
        selected_numbers = None
        
        for attempt in range(max_attempts):
            # 1. 从Top10中随机抽取3个胆码
            anchors = random.sample(top10_hot, 3)
            
            # 2. 从剩余热池中按概率抽取3个拖码
            # 排除胆码
            temp_remaining_hot = [n for n in remaining_hot_pool if n not in anchors]
            hot_selected = select_numbers_from_pool(
                temp_remaining_hot, all_scores, 3
            )
            
            # 3. 从冷池中按概率抽取1个拖码（排除胆码）
            temp_cold_pool = [n for n in cold_pool if n not in anchors]
            if temp_cold_pool:
                cold_selected = select_numbers_from_pool(
                    temp_cold_pool, all_scores, 1
                )
            else:
                cold_selected = []
            
            # 合并
            selected = sorted(anchors + hot_selected + cold_selected)
            
            # 验证和值
            if len(selected) == num_count and is_sum_valid(selected, target_sum, tolerance):
                selected_numbers = selected
                break
        
        # 如果没找到符合和值的，用最后一次生成的
        if selected_numbers is None:
            anchors = random.sample(top10_hot, 3)
            temp_remaining_hot = [n for n in remaining_hot_pool if n not in anchors]
            hot_selected = select_numbers_from_pool(temp_remaining_hot, all_scores, 3)
            temp_cold_pool = [n for n in cold_pool if n not in anchors]
            if temp_cold_pool:
                cold_selected = select_numbers_from_pool(temp_cold_pool, all_scores, 1)
            else:
                cold_selected = []
            selected_numbers = sorted(anchors + hot_selected + cold_selected)
        
        total = sum(selected_numbers)
        
        bets.append({
            'numbers': selected_numbers,
            'sum': total,
            'target': f'方法B(目标{target_sum})',
            'deviation': total - target_sum,
            'anchors': anchors
        })
    
    return bets
# --------------------
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
        draws = load_recent_draws_from_supabase(limit=500)
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
#--------
# 创建一个占位符，用于显示更新结果（整行宽度）
update_placeholder = st.empty()

#------------
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.5])

with col1:
    st.metric("最新期次", latest_draw.get('period', 'N/A'))
with col2:
    latest_date = latest_draw.get('date', 'N/A')
    if latest_date and len(latest_date) > 10:
        latest_date = latest_date[:10]
    st.metric("最新日期", latest_date)
with col3:
    st.metric("最早期次", oldest_draw.get('period', 'N/A'))
with col4:
    st.metric("数据总量", f"{len(draws)} 期")
with col5:
    st.write("")
    if st.button("🔄 检查更新", key="update_btn", help="从网站抓取最新开奖数据"):
        # 清空旧的占位符消息
        update_placeholder.empty()
        with st.spinner("正在检查更新..."):
            # 1. 获取当前数据库最新期次
            current_draws = load_draws_from_supabase()
            if current_draws:
                sorted_draws = sorted(current_draws, key=lambda x: int(x.get('period', 0)), reverse=True)
                latest_db_period = sorted_draws[0]['period']
            else:
                latest_db_period = 0

            # 2. 抓取网页数据
            new_data = fetch_latest_from_9800()
            if not new_data:
                update_placeholder.warning("未获取到数据，请检查网络或稍后重试")
            else:
                # 3. 筛选出比数据库最新期次更大的数据
                to_insert = [d for d in new_data if d['period'] > latest_db_period]
                if not to_insert:
                    update_placeholder.info(f"✅ 已是最新数据（最新期次: {latest_db_period}）")
                else:
                    # 4. 插入新数据
                    try:
                        supabase = init_supabase()
                        if supabase is None:
                            update_placeholder.error("Supabase 连接失败")
                        else:
                            # 批量插入（使用 upsert 避免重复）
                            for rec in to_insert:
                                record = {
                                    "period": rec['period'],
                                    "date": rec['date'],
                                    "numbers": rec['numbers'],
                                    "special": rec['special'],
                                    "sum_7": rec['sum'],
                                    "sum_value": rec['sum']
                                }
                                supabase.schema('marksix_schema').table('marksix_draws').upsert(
                                    record, on_conflict='period'
                                ).execute()
                            
                            # 5. 检查总行数，若超过1000则删除最旧的数据
                            total_count = supabase.schema('marksix_schema').table('marksix_draws').select("*", count="exact").execute()
                            count = total_count.count if hasattr(total_count, 'count') else len(total_count.data)
                            if count > 1000:
                                # 获取最旧的 (count - 1000) 条的 period
                                old_records = supabase.schema('marksix_schema').table('marksix_draws')\
                                    .select("period").order("period", desc=False).limit(count - 1000).execute()
                                old_periods = [r['period'] for r in old_records.data]
                                if old_periods:
                                    for p in old_periods:
                                        supabase.schema('marksix_schema').table('marksix_draws')\
                                            .delete().eq("period", p).execute()
                            
                            update_placeholder.success(f"✅ 成功新增 {len(to_insert)} 期数据，已保留最新1000期")
                            # 刷新页面数据
                            st.session_state['draws_loaded'] = load_draws_from_supabase()
                            st.rerun()
                    except Exception as e:
                        update_placeholder.error(f"更新失败: {e}")
        
st.markdown("---")
# ===
# ==================== 冷热码分析 & 方法A详情 ====================
tab1, tab2 = st.tabs(["📊 冷热码分析", "🎯 方法A：分池评分"])

with tab1:
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
#-----
class MethodAConfig:
    """方法A配置类，存储所有可调参数"""
    hot_count = 6
    cold_count = 1
    hot_range = (0, 10)
    hot_temperature = 0.8
    cold_temperature = 0.8
    zone_window = 15
#-----
def get_method_a_config_from_session():
    """从session_state获取方法A配置"""
    config = MethodAConfig()
    config.hot_count = st.session_state.get('method_a_hot_count', 6)
    config.cold_count = st.session_state.get('method_a_cold_count', 1)
    config.hot_range = (st.session_state.get('method_a_hot_range_start', 0), 
                        st.session_state.get('method_a_hot_range_end', 10))
    config.hot_temperature = st.session_state.get('method_a_hot_temperature', 0.8)
    config.cold_temperature = st.session_state.get('method_a_cold_temperature', 0.8)
    config.zone_window = st.session_state.get('method_a_zone_window', 15)
    return config
#-----
def show_method_a_score_details(draws):
    """显示方法A的详细评分信息 - 三列并排版"""
    st.markdown("### 📊 方法A：分池评分详情")
    
    if not draws:
        st.warning("请先加载数据")
        return
    
    config = get_method_a_config_from_session()
    
    zone_bonus_config = {
        1: st.session_state.get('zone1_bonus', 8),
        2: st.session_state.get('zone2_bonus', 5),
        3: st.session_state.get('zone3_bonus', 3),
        4: st.session_state.get('zone4_bonus', 0),
        5: st.session_state.get('zone5_bonus', 0),
        6: st.session_state.get('zone6_bonus', 0),
        7: st.session_state.get('zone7_bonus', 0),
    }
    
    pattern_config = {
        "gap_2": st.session_state.get('pattern_gap2', 15),
        "gap_3": st.session_state.get('pattern_gap3', 10),
        "edge_normal": st.session_state.get('pattern_edge_normal', 8),
        "edge_special": st.session_state.get('pattern_edge_special', 6),
        "consecutive": st.session_state.get('pattern_consecutive', 5),
        "alternate": st.session_state.get('pattern_alternate', 5),
        "max": st.session_state.get('pattern_max', 25),
    }
    
    cold_config = {
        "frequency_acceleration": st.session_state.get('cold_freq_acc', 12),
        "miss_13_15": st.session_state.get('cold_miss_13_15', 8),
        "consecutive": st.session_state.get('cold_consecutive', 10),
        "cold_return": st.session_state.get('cold_return', 8),
        "cold_neighbor": st.session_state.get('cold_neighbor', 5),
        "max": st.session_state.get('cold_max', 20),
    }
    
    with st.spinner("计算评分中..."):
        details = get_method_a_score_details(
            draws,
            hot_range=config.hot_range,
            zone_window=config.zone_window,
            zone_bonus_config=zone_bonus_config,
            pattern_config=pattern_config,
            cold_config=cold_config,
            hot_temperature=config.hot_temperature,
            cold_temperature=config.cold_temperature
        )
    
    # 显示当前配置信息
    st.info(f"🎯 当前池间分配: 热池{config.hot_count}个 + 冷池{config.cold_count}个 | 热池温度: {config.hot_temperature} | 冷池温度: {config.cold_temperature}")
    
    st.markdown("---")
    
    # ========== 三列并排布局 ==========
    col_left, col_center, col_right = st.columns(3)
    
    # 左列：热池Top20
    with col_left:
        st.markdown("### 🔥 热池Top20（遗漏0-10期）")
        
        hot_scores_with_prob = [(num, details['scores'][num], details['hot_probs'].get(num, 0)) 
                                for num in details['hot_pool']]
        hot_scores_with_prob.sort(key=lambda x: x[1], reverse=True)
        
        hot_df = pd.DataFrame([
            {"号码": num, "评分": score, "概率": f"{prob*100:.2f}%"}
            for num, score, prob in hot_scores_with_prob[:20]
        ])
        st.dataframe(hot_df, use_container_width=True, hide_index=True)
    
    # 中列：冷池Top20
    with col_center:
        st.markdown("### ❄️ 冷池Top20（遗漏>10期）")
        
        cold_scores_with_prob = [(num, details['scores'][num], details['cold_probs'].get(num, 0)) 
                                for num in details['cold_pool']]
        cold_scores_with_prob.sort(key=lambda x: x[1], reverse=True)
        
        cold_df = pd.DataFrame([
            {"号码": num, "评分": score, "概率": f"{prob*100:.2f}%"}
            for num, score, prob in cold_scores_with_prob[:20]
        ])
        st.dataframe(cold_df, use_container_width=True, hide_index=True)
    
    # 右列：当前热区7分区
    with col_right:
        st.markdown("### 🗺️ 当前热区7分区")
        
        zone_data = []
        for zone in range(1, 8):
            zone_name = f"{chr(64+zone)}区"
            zone_range = f"{get_zone_numbers(zone)[0]:02d}-{get_zone_numbers(zone)[-1]:02d}"
            count = details['zone_counts'][zone]
            rank = details['zone_rank'][zone]
            bonus = zone_bonus_config.get(rank, 0)
            zone_data.append({
                "分区": zone_name,
                "范围": zone_range,
                "出现次数": count,
                "排名": f"第{rank}热区" if rank <= 3 else f"第{rank}区",
                "加分": f"+{bonus}" if bonus > 0 else "0"
            })
        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)
#-----
with tab2:
    # 方法A分池评分详情
    show_method_a_score_details(draws)

st.markdown("---")
#-----
# ==================== 和值趋势分析 ====================
st.subheader("📈 和值趋势分析")

show_periods = st.slider(
    "显示最近期数", 
    min_value=10, 
    max_value=min(200, len(draws)), 
    value=min(100, len(draws)), 
    step=10
)

# 获取最新数据（按时间顺序，旧到新）
recent_draws = draws[-show_periods:] if len(draws) >= show_periods else draws
periods = [str(d.get('period', '')) for d in recent_draws]
sum_7_values = [d.get('sum', sum(d.get('numbers', []))) for d in recent_draws]

# 创建DataFrame
sum_df = pd.DataFrame({
    '期次': periods,
    '和值(7码)': sum_7_values,
    '序号': list(range(len(periods)))
})

# 计算正弦拟合线（基于最近10期）
y_fit = []
fit_x = []
next_val = None
if len(recent_draws) >= 10:
    recent_10_sums = sum_7_values[-10:]
    try:
        from scipy.optimize import curve_fit
        def sine_func(x, A, omega, phi, C):
            return A * np.sin(omega * x + phi) + C
        x = np.arange(10)
        y = np.array(recent_10_sums)
        A_guess = (np.max(y) - np.min(y)) / 2
        C_guess = np.mean(y)
        omega_guess = 2 * np.pi / 6.5
        params, _ = curve_fit(sine_func, x, y, p0=[A_guess, omega_guess, 0, C_guess], maxfev=2000)
        A, omega, phi, C = params
        x_pred = np.arange(11)
        y_pred = sine_func(x_pred, A, omega, phi, C)
        y_fit = y_pred[:-1]
        fit_x = list(range(len(sum_df) - 10, len(sum_df)))
        next_val = y_pred[-1]
    except:
        pass

# 导入 altair
import altair as alt

# 绘制实际和值
chart = alt.Chart(sum_df).mark_line(point=True, color='darkblue').encode(
    x=alt.X('序号:Q', title='期次', axis=alt.Axis(labelAngle=-45, tickCount=10)),
    y=alt.Y('和值(7码):Q', title='和值(7码)'),
    tooltip=['期次', '和值(7码)']
).properties(
    title=f'最近{show_periods}期和值走势',
    height=400
)

# 添加正弦拟合线
if len(y_fit) > 0:
    fit_df = pd.DataFrame({
        '序号': fit_x,
        '拟合值': y_fit
    })
    fit_line = alt.Chart(fit_df).mark_line(color='#ff7f0e', strokeDash=[5, 5]).encode(
        x='序号:Q',
        y='拟合值:Q'
    )
    chart = chart + fit_line

# 添加理论均值线
mean_line = alt.Chart(pd.DataFrame({'y': [175]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')

st.altair_chart(chart + mean_line, use_container_width=True)

st.markdown("**📊 和值统计**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("理论均值", "175")
with col2:
    st.metric("历史均值", f"{np.mean(sum_7_values):.1f}")
with col3:
    current_sum = sum_7_values[-1] if sum_7_values else 175
    st.metric("当前和值", f"{current_sum}")

st.markdown("---")
st.markdown("**🎯 和值预测参考**")

# 计算四种方法的预测范围
# 1. 动态回归
dynamic_target, dynamic_tolerance, _, _, _, _, _ = get_dynamic_sum_range(draws, num_count=7, window=4)
dynamic_lower = max(140, dynamic_target - dynamic_tolerance)
dynamic_upper = min(210, dynamic_target + dynamic_tolerance)

# 2. 均值回归
mean_lower, mean_upper = get_target_sum_mean_reversion_range(draws)

# 3. 移动平均
ma_lower, ma_upper = get_target_sum_moving_average_range(draws)

# 4. 正弦拟合
sine_lower, sine_upper = get_target_sum_sine_range(draws)

# 显示四个指标
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("动态回归", f"{dynamic_lower}-{dynamic_upper}", delta=f"±{dynamic_tolerance}")
with col2:
    st.metric("均值回归", f"{mean_lower}-{mean_upper}", delta="±17")
with col3:
    st.metric("移动平均(7期)", f"{ma_lower}-{ma_upper}", delta="±17")
with col4:
    st.metric("正弦拟合", f"{sine_lower}-{sine_upper}", delta="±17")

# 和值预测方法选择（添加动态回归选项）
sum_predict_method = st.radio(
    "选择预测方法（用于选号）",
    options=["动态回归", "均值回归", "移动平均(7期)", "正弦拟合"],
    index=2,  # 默认动态回归
    key="sum_predict_method",
    horizontal=True
)

st.markdown("---")
# ===
# ==================== 方法A：分池评分法 - 配置和辅助函数 ====================

def get_method_a_config_from_session():
    """从session_state获取方法A配置"""
    config = MethodAConfig()
    config.hot_count = st.session_state.get('method_a_hot_count', 6)
    config.cold_count = st.session_state.get('method_a_cold_count', 1)
    config.hot_range = (st.session_state.get('method_a_hot_range_start', 0), 
                        st.session_state.get('method_a_hot_range_end', 10))
    config.hot_temperature = st.session_state.get('method_a_hot_temperature', 0.8)
    config.cold_temperature = st.session_state.get('method_a_cold_temperature', 0.8)
    config.zone_window = st.session_state.get('method_a_zone_window', 15)
    return config

# ===========
def generate_method_a_bets_wrapper(draws, num_bets, num_count, random_seed, sum_predict_method):
    """方法A投注生成包装函数"""
    config = get_method_a_config_from_session()
    
    zone_bonus_config = {
        1: st.session_state.get('zone1_bonus', 8),
        2: st.session_state.get('zone2_bonus', 5),
        3: st.session_state.get('zone3_bonus', 3),
        4: st.session_state.get('zone4_bonus', 0),
        5: st.session_state.get('zone5_bonus', 0),
        6: st.session_state.get('zone6_bonus', 0),
        7: st.session_state.get('zone7_bonus', 0),
    }
    
    pattern_config = {
        "gap_2": st.session_state.get('pattern_gap2', 15),
        "gap_3": st.session_state.get('pattern_gap3', 10),
        "edge_normal": st.session_state.get('pattern_edge_normal', 8),
        "edge_special": st.session_state.get('pattern_edge_special', 6),
        "consecutive": st.session_state.get('pattern_consecutive', 5),
        "alternate": st.session_state.get('pattern_alternate', 5),
        "max": st.session_state.get('pattern_max', 25),
    }
    
    cold_config = {
        "frequency_acceleration": st.session_state.get('cold_freq_acc', 12),
        "miss_13_15": st.session_state.get('cold_miss_13_15', 8),
        "consecutive": st.session_state.get('cold_consecutive', 10),
        "cold_return": st.session_state.get('cold_return', 8),
        "cold_neighbor": st.session_state.get('cold_neighbor', 5),
        "max": st.session_state.get('cold_max', 20),
    }
    
    return generate_bets_method_a(
        draws, num_bets, num_count,
        hot_count=config.hot_count, cold_count=config.cold_count,
        hot_range=config.hot_range,
        hot_temperature=config.hot_temperature,
        cold_temperature=config.cold_temperature,
        zone_window=config.zone_window,
        zone_bonus_config=zone_bonus_config,
        pattern_config=pattern_config,
        cold_config=cold_config,
        sum_predict_method=sum_predict_method,
        random_seed=random_seed
    )

# ===========
# ===========
def show_method_a_advanced_settings():
    """显示方法A的高级设置"""
    with st.expander("⚙️ 方法A高级设置（可调整评分参数）", expanded=False):
        st.markdown("**📊 池间分配**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("热池抽取个数", min_value=4, max_value=6, value=6, step=1, key="method_a_hot_count")
        with col2:
            st.number_input("冷池抽取个数", min_value=1, max_value=3, value=1, step=1, key="method_a_cold_count")
        
        st.markdown("---")
        st.markdown("**🔥 热池参数**")
        col1, col2 = st.columns(2)
        with col1:
            # 温度参数已废弃，保留注释
            pass
        with col2:
            st.number_input("热池遗漏起始", min_value=0, max_value=5, value=0, step=1, key="method_a_hot_range_start")
            st.number_input("热池遗漏结束", min_value=5, max_value=15, value=10, step=1, key="method_a_hot_range_end")
        
        st.markdown("---")
        st.markdown("**❄️ 冷池参数**")
        col1, col2 = st.columns(2)
        with col1:
            # 温度参数已废弃，保留注释
            pass
        with col2:
            st.number_input("冷池遗漏起始", min_value=8, max_value=15, value=11, step=1, key="method_a_cold_range_start")
        
        st.markdown("---")
        st.markdown("**🗺️ 分区热度参数**")
        st.slider("分区窗口期（期数）", min_value=10, max_value=30, value=15, step=5, key="method_a_zone_window")
        
        st.markdown("**分区加分（可自定义）**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("第1热区加分", value=8, min_value=0, max_value=20, key="zone1_bonus")
            st.number_input("第2热区加分", value=5, min_value=0, max_value=20, key="zone2_bonus")
        with col2:
            st.number_input("第3热区加分", value=3, min_value=0, max_value=20, key="zone3_bonus")
            st.number_input("第4热区加分", value=0, min_value=0, max_value=20, key="zone4_bonus")
        with col3:
            st.number_input("第5热区加分", value=0, min_value=0, max_value=20, key="zone5_bonus")
            st.number_input("第6热区加分", value=0, min_value=0, max_value=20, key="zone6_bonus")
            st.number_input("第7热区加分", value=0, min_value=0, max_value=20, key="zone7_bonus")
        
        st.markdown("**📐 规律加分（可自定义）**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("夹号-间隔2", value=15, min_value=0, max_value=25, key="pattern_gap2")
            st.number_input("夹号-间隔3", value=10, min_value=0, max_value=20, key="pattern_gap3")
        with col2:
            st.number_input("边号(正码)", value=8, min_value=0, max_value=15, key="pattern_edge_normal")
            st.number_input("边号(特码)", value=6, min_value=0, max_value=15, key="pattern_edge_special")
        with col3:
            st.number_input("连号潜力", value=5, min_value=0, max_value=10, key="pattern_consecutive")
            st.number_input("隔期模式", value=5, min_value=0, max_value=10, key="pattern_alternate")
            st.number_input("单号上限", value=25, min_value=15, max_value=35, key="pattern_max")
        
        st.markdown("**🔄 冷码专有加分**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("频率加速度", value=12, min_value=0, max_value=20, key="cold_freq_acc")
            st.number_input("遗漏13-15期", value=8, min_value=0, max_value=15, key="cold_miss_13_15")
        with col2:
            st.number_input("连续2期出现", value=10, min_value=0, max_value=15, key="cold_consecutive")
            st.number_input("冷号回补", value=8, min_value=0, max_value=15, key="cold_return")
        with col3:
            st.number_input("相邻冷号", value=5, min_value=0, max_value=10, key="cold_neighbor")
            st.number_input("冷码上限", value=20, min_value=10, max_value=30, key="cold_max")
        
        if st.button("恢复默认设置", key="reset_method_a"):
            for key in list(st.session_state.keys()):
                if key.startswith("method_a_") or key in ["zone1_bonus", "zone2_bonus", "zone3_bonus", 
                    "zone4_bonus", "zone5_bonus", "zone6_bonus", "zone7_bonus", "pattern_gap2", "pattern_gap3",
                    "pattern_edge_normal", "pattern_edge_special", "pattern_consecutive", "pattern_alternate",
                    "pattern_max", "cold_freq_acc", "cold_miss_13_15", "cold_consecutive", "cold_return",
                    "cold_neighbor", "cold_max"]:
                    del st.session_state[key]
            st.rerun()

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
            "方法A: 分池评分法 ⭐推荐",
            "方法B: 新胆拖混合（基于方法A评分）",
            "方法1: 当前方法",
            "方法2: 胆拖混合",
            "方法3: LightGBM",
            "方法4: XGBoost+NN",
            "方法5: 综合模式"
        ],
        index=0,
        key="ai_model"
    )

with st.expander("⚙️ 高级设置"):
    # ==================== 方法A高级设置（可折叠） ====================
    show_method_a_advanced_settings()
    
    st.markdown("---")
    st.markdown("**📈 通用参数**")
    
    col1, col2 = st.columns(2)
    with col1:
        trend_window = st.number_input("和值趋势窗口", min_value=2, max_value=20, value=4, step=1, key="trend_window")
    with col2:
        st.markdown("**🎲 随机种子模式**")
        seed_mode = st.radio(
            "选择种子模式",
            options=["日期+时间", "用户输入固定种子", "机器自动产生（每期随机）"],
            index=0,
            key="seed_mode",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        seed_date = None
        seed_time = None
        fixed_seed_value = None
        
        if seed_mode == "日期+时间":
            col_date, col_time = st.columns(2)
            with col_date:
                seed_date = st.date_input("选择日期", value=datetime.now().date(), key="seed_date")
            with col_time:
                seed_time = st.time_input("选择时间", value=datetime.strptime("21:30", "%H:%M").time(), key="seed_time")
        elif seed_mode == "用户输入固定种子":
            fixed_seed_value = st.number_input("输入固定种子值", min_value=0, max_value=1000000, value=7, step=1, key="fixed_seed_value")
        # 机器自动产生：不需要额外输入
    #----------------
    st.markdown("**📊 训练期数设置**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        method_b_window = st.number_input("方法B期数", min_value=10, max_value=300, value=20, step=5, key="m_b_window")
    with col2:
        method_a_window = st.number_input("方法A期数", min_value=10, max_value=300, value=20, step=5, key="m_a_window")
    with col3:
        method1_window = st.number_input("方法1/2期数", min_value=10, max_value=300, value=20, step=5, key="m1_window")
    with col4:
        method3_window = st.number_input("方法3期数", min_value=10, max_value=300, value=20, step=5, key="m3_window")
    with col5:
        method4_window = st.number_input("方法4/5期数", min_value=10, max_value=300, value=20, step=5, key="m4_window")

# 显示和值预测信息
# 根据用户选择的预测方法获取和值范围
if sum_predict_method == "均值回归":
    sum_lower, sum_upper = get_target_sum_mean_reversion_range(draws)
    sum_method_name = "均值回归"
elif sum_predict_method == "移动平均(7期)":
    sum_lower, sum_upper = get_target_sum_moving_average_range(draws)
    sum_method_name = "移动平均(7期)"
else:
    sum_lower, sum_upper = get_target_sum_sine_range(draws)
    sum_method_name = "正弦拟合"

st.caption(f"💡 **和值预测 ({sum_method_name})**: 范围 {sum_lower}-{sum_upper}")
#------------------
if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    # 解析随机种子
    random_seed = None
    if seed_mode == "日期+时间":
        if seed_date and seed_time:
            dt = datetime.combine(seed_date, seed_time)
            random_seed = int(dt.timestamp())
            st.success(f"✅ 已设置随机种子: {seed_date} {seed_time}")
        else:
            st.warning("⚠️ 请选择日期和时间")
    elif seed_mode == "用户输入固定种子":
        if fixed_seed_value is not None:
            random_seed = int(fixed_seed_value)
            st.success(f"✅ 已设置固定种子: {fixed_seed_value}")
        else:
            st.warning("⚠️ 请输入固定种子值")
    else:  # 机器自动产生
        random_seed = None
        st.info("🔧 使用机器自动产生的随机种子（每期不同）")
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        # ========== 读取智能投注独立的训练窗口值 ==========
        # 这些值在高级设置中已通过 st.number_input 定义
        # method_b_window, method_a_window, method1_window, method3_window, method4_window
        # ================================================
        
        if "方法A" in ai_model:
            # 方法A使用 method_a_window 截断数据
            limited_draws = draws[-method_a_window:] if len(draws) > method_a_window else draws
            bets = generate_method_a_bets_wrapper(
                limited_draws, num_bets, num_count, random_seed, sum_predict_method
            )
            model_used = "方法A: 分池评分法"
            
        elif "方法B" in ai_model:
            # 方法B使用 method_b_window 截断数据
            limited_draws = draws[-method_b_window:] if len(draws) > method_b_window else draws
            bets = generate_bets_method_b(
                limited_draws, num_bets, num_count, sum_predict_method, random_seed
            )
            model_used = "方法B: 新胆拖混合（基于方法A评分）"
            
        elif "方法1" in ai_model:
            # 方法1使用 method1_window 截断数据
            limited_draws = draws[-method1_window:] if len(draws) > method1_window else draws
            bets = generate_bets_method1_current(
                limited_draws, num_bets, num_count, trend_window, random_seed, method1_window, sum_predict_method
            )
            model_used = "方法1: 当前方法"
            
        elif "方法2" in ai_model:
            # 方法2使用 method1_window 截断数据（与方法1共用）
            limited_draws = draws[-method1_window:] if len(draws) > method1_window else draws
            bets = generate_bets_method2_hybrid(
                limited_draws, num_bets, num_count, trend_window, random_seed, method1_window, sum_predict_method
            )
            model_used = "方法2: 胆拖混合"
            
        elif "方法3" in ai_model:
            # 方法3使用 method3_window 截断数据
            limited_draws = draws[-method3_window:] if len(draws) > method3_window else draws
            bets = generate_bets_method3_lightgbm(
                limited_draws, num_bets, num_count, trend_window, random_seed, method3_window, sum_predict_method
            )
            model_used = "方法3: LightGBM"
            
        elif "方法4" in ai_model:
            # 方法4使用 method4_window 截断数据
            # 截断后的数据传入 XGBoost，配合 build_advanced_features 中 30→15 的修改，可正常训练
            limited_draws = draws[-method4_window:] if len(draws) > method4_window else draws
            bets = generate_bets_method4_ensemble(
                limited_draws, num_bets, num_count, trend_window, random_seed, method4_window, sum_predict_method
            )
            model_used = "方法4: XGBoost+NN"
            
        else:
            # 方法5：综合模式，传入所有窗口值
            limited_draws = draws[-method4_window:] if len(draws) > method4_window else draws
            bets = generate_bets_method5_ensemble(
                limited_draws, num_bets, num_count, trend_window, random_seed,
                method1_window, method1_window, method3_window, method4_window
            )
            model_used = "方法5: 综合模式"
    
    st.session_state['generated_bets'] = bets
    st.session_state['model_used'] = model_used
    st.success(f"✅ 使用 {model_used} 生成 {len(bets)} 组{num_count}码复式")
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
st.caption("📌 粘贴实际开奖数据（最多5期），下方 Tab 选择比对来源")

# ---------- 公用区域：粘贴多期开奖数据 ----------
check_draws_text = st.text_area(
    "📋 粘贴多期开奖数据（最多5期）",
    height=120,
    key="check_draws",
    placeholder="示例:\n26045 2026-04-25 4 16 21 36 42 46 9\n26044 2026-04-23 12 23 37 38 45 48 8"
)

# ========== 辅助函数：解析用户自定义投注 ==========
def parse_custom_bets(text: str) -> List[List[int]]:
    """解析用户输入的自定义投注，每行一注，返回号码列表的列表"""
    bets = []
    lines = text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        # 使用正则分割：支持空格、制表符、逗号、中文逗号、顿号等
        parts = re.split(r'[,\s\t，、]+', line.strip())
        parts = [p for p in parts if p]  # 过滤空字符串
        nums = []
        valid = True
        for p in parts:
            try:
                num = int(p)
                if 1 <= num <= 49:
                    nums.append(num)
                else:
                    valid = False
                    break
            except ValueError:
                valid = False
                break
        # 去重并排序，检查数量（6-49个）
        nums = sorted(set(nums))
        if valid and 6 <= len(nums) <= 49:
            bets.append(nums)
    return bets

# ========== 辅助函数：计算匹配分数（通用版，适用于任何号码集合） ==========
def calculate_match_score_for_draws(bet_numbers: List[int], check_draws: List[Dict]) -> List[str]:
    """
    计算投注在每期开奖中的匹配分数
    规则：命中正码个数 + (命中特码 ? 0.5 : 0)
    """
    results = []
    bet_set = set(bet_numbers)
    for draw in check_draws:
        draw_numbers = set(draw['numbers'])  # 6个正码
        draw_special = draw.get('special')   # 特码
        
        # 命中正码个数
        main_hits = len(bet_set & draw_numbers)
        # 是否命中特码（加0.5）
        special_hit = 0.5 if draw_special in bet_set else 0
        score = main_hits + special_hit
        
        if score == int(score):
            results.append(str(int(score)))
        else:
            results.append(f"{score:.1f}")
    return results

# 为了与旧代码兼容，保留原名（实际上已统一）
# calculate_match_score_for_draws 已经定义，不需要别名

# ========== Tab 切换 ==========
tab_ai, tab_custom = st.tabs(["🤖 AI推荐组合", "✏️ 自定义组合"])

# ==================== Tab 1: AI推荐组合（原逻辑保留） ====================
with tab_ai:
    st.caption("📌 使用当前智能投注生成的组合进行查奖")
    
    if st.button("🔍 查奖（AI组合）", key="check_btn_ai"):
        check_draws = parse_multi_draws_for_checking(check_draws_text, max_draws=5)
        if not check_draws:
            st.error("解析失败，请检查格式")
        elif not st.session_state.get('generated_bets'):
            st.warning("请先生成投注组合")
        else:
            st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
            
            # ===== 原 AI 查奖逻辑（完全保留，计分函数已统一） =====
            enhanced_bets_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                numbers_display = ','.join(f"{n:02d}" for n in bet['numbers'])
                row = {'组别': i, f'{len(bet["numbers"])}个号码': numbers_display, '和值': bet['sum']}
                
                # 调用统一的计分函数
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

# ==================== Tab 2: 自定义组合（全新功能） ====================
with tab_custom:
    st.caption("📝 每行输入一注复式号码（6-49个），用空格或逗号分隔")
    
    # 自定义号码输入框（高度约150px，约6-8行）
    custom_bets_text = st.text_area(
        "输入投注组合",
        height=150,
        key="custom_bets_input",
        placeholder="例:\n1,5,12,23,30,42,47\n2 8 15 24 31 40 45\n3,9,16,25,32,41,46,7"
    )
    
    if st.button("🔍 查奖（自定义组合）", key="check_btn_custom"):
        # 1. 解析开奖数据
        check_draws = parse_multi_draws_for_checking(check_draws_text, max_draws=5)
        if not check_draws:
            st.error("请先在顶部粘贴正确的开奖数据（至少1期）")
        elif not custom_bets_text.strip():
            st.warning("请至少输入一组投注号码")
        else:
            # 2. 解析用户输入的每一注
            custom_bets = parse_custom_bets(custom_bets_text)
            
            if not custom_bets:
                st.error("没有有效的投注组合（请确保每行6-49个1-49的不重复数字）")
            else:
                st.success(f"✅ 成功解析 {len(custom_bets)} 组有效投注，比对 {len(check_draws)} 期数据")
                
                # 3. 构建结果表格
                result_data = []
                for idx, bet_nums in enumerate(custom_bets, 1):
                    # 投注号码列：完整显示所有号码
                    numbers_display = ','.join(f"{n:02d}" for n in bet_nums)
                    row = {
                        "组别": idx,
                        "投注号码": numbers_display
                    }
                    
                    # 计算每一期的得分（复用同一函数）
                    match_scores = calculate_match_score_for_draws(bet_nums, check_draws)
                    for j, draw in enumerate(check_draws):
                        period_str = str(draw['period'])
                        if len(period_str) > 10:
                            period_str = period_str[-10:]
                        row[f'中奖_{period_str}'] = match_scores[j]
                    
                    result_data.append(row)
                
                # 4. 展示结果（带横向滚动）
                df_result = pd.DataFrame(result_data)
                
                # 设置列宽：投注号码列宽大，其余紧凑
                column_config = {}
                for col in df_result.columns:
                    if col == "投注号码":
                        column_config[col] = st.column_config.TextColumn(
                            "投注号码",
                            width="large",
                            help="完整显示所有号码"
                        )
                    elif col == "组别":
                        column_config[col] = st.column_config.NumberColumn("组别", width="small")
                    else:
                        column_config[col] = st.column_config.TextColumn(col, width="small")
                
                st.dataframe(
                    df_result,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                # 显示开奖数据预览
                preview_df = pd.DataFrame([
                    {'期次': d['period'], '正码': str(d['numbers']), '特码': d['special']}
                    for d in check_draws
                ])
                st.markdown("**📊 开奖数据预览**")
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                
                # 计分说明
                st.caption("💡 得分 = 命中正码个数 + (命中特码 ? 0.5 : 0)")
# ==================== 策略回测（5种方法对比） ====================
st.markdown("---")
st.subheader("📊 策略回测（5种方法对比）")
st.caption("测试不同策略在历史数据上的表现（基于当前Supabase中的数据）")

# 获取数据
backtest_draws = load_draws_from_supabase()

if backtest_draws is None or len(backtest_draws) < 50:
    st.warning("⚠️ 数据不足，至少需要50期数据才能进行回测")
else:
    sorted_backtest_draws = sorted(backtest_draws, key=lambda x: int(x.get('period', 0)) if str(x.get('period', 0)).isdigit() else 0)
    total_draws_count = len(backtest_draws)
    st.info(f"📊 当前云端有 {total_draws_count} 期数据 (范围: {sorted_backtest_draws[0].get('period')} - {sorted_backtest_draws[-1].get('period')})")
    
    # ========== 回测参数设置 ==========
    with st.expander("⚙️ 回测参数设置", expanded=False):
        st.markdown("**📊 回测方法选择**")
        col_methodB, col_methodA, col_method1, col_method2, col_method3, col_method4, col_method5 = st.columns(7)
        with col_methodB:
            enable_method_b = st.checkbox("方法B", value=True, key="bt_enable_method_b")
        with col_methodA:
            enable_method_a = st.checkbox("方法A", value=True, key="bt_enable_method_a")
        with col_method1:
            enable_method1 = st.checkbox("方法1", value=True, key="bt_enable_method1")
        with col_method2:
            enable_method2 = st.checkbox("方法2", value=True, key="bt_enable_method2")
        with col_method3:
            enable_method3 = st.checkbox("方法3", value=True, key="bt_enable_method3")
        with col_method4:
            enable_method4 = st.checkbox("方法4", value=True, key="bt_enable_method4")
        with col_method5:
            enable_method5 = st.checkbox("方法5", value=True, key="bt_enable_method5")
        
        st.markdown("---")
        st.markdown("**📈 回测参数**")
        #-----------
        # 第一行：两列并排（每注号码数、和值趋势窗口）
        col1, col2 = st.columns(2)
        with col1:
            test_num_count = st.selectbox("每注号码数", [6, 7, 8, 9, 10], index=1, key="backtest_num_count")
            test_bets = st.number_input("每期组数", min_value=1, max_value=20, value=4, step=1, key="backtest_bets")
        with col2:
            test_trend_window = st.number_input("和值趋势窗口", min_value=2, max_value=20, value=4, step=1, key="backtest_trend_window")
        
        # 第二行：训练期数设置（四列并排）
        st.markdown("**📊 训练期数设置**")
        col_b, col_a, col_1, col_3, col_4 = st.columns(5)
        with col_b:
            method_b_window = st.number_input("方法B期数", min_value=20, max_value=500, value=20, step=5, key="bt_method_b_window")
        with col_a:
            method_a_window = st.number_input("方法A期数", min_value=20, max_value=500, value=20, step=5, key="bt_method_a_window")
        with col_1:
            method1_window = st.number_input("方法1/2期数", min_value=20, max_value=200, value=20, step=5, key="bt_method1_window")
        with col_3:
            method3_window = st.number_input("方法3 LightGBM期数", min_value=10, max_value=300, value=20, step=5, key="bt_method3_window")
        with col_4:
            method4_window = st.number_input("方法4/5 XGBoost+NN期数", min_value=10, max_value=300, value=20, step=5, key="bt_method4_window")
        
        # 第三行：回测期数设置
        st.markdown("**📈 回测期数设置**")
        max_window = max(method_b_window, method_a_window, method1_window, method3_window, method4_window)
        max_backtest_periods = total_draws_count - max_window
        if max_backtest_periods < 1:
            st.error(f"数据不足：需要至少{max_window}期数据，当前只有{total_draws_count}期")
            st.stop()
        
        test_periods = st.number_input(
            "回测期数", 
            min_value=1, 
            max_value=max_backtest_periods, 
            value=min(100, max_backtest_periods), 
            key="backtest_periods"
        )
        st.caption(f"📌 最大可用回测期数: {max_backtest_periods}期")
    
    st.markdown("---")
    st.markdown("**🎲 随机种子模式**")
    col_seed1, col_seed2 = st.columns(2)
    with col_seed1:
        seed_mode_option = st.radio(
            "选择种子模式",
            options=["日期+时间（每期用自己的开奖日期+21:30）", "用户输入固定种子", "机器自动产生（每期随机）"],
            index=0,
            key="backtest_seed_mode",
            horizontal=False
        )
    with col_seed2:
        fixed_seed_value = 1
        if "用户输入固定种子" in seed_mode_option:
            fixed_seed_value = st.number_input("请输入固定种子值", min_value=0, max_value=10000, value=7, step=1, key="bt_fixed_seed")
        
        # 显示当前设置说明
        if "日期+时间" in seed_mode_option:
            st.caption("📅 每期使用开奖日期 + 21:30 生成种子")
    
    # 映射种子模式
    if "日期+时间" in seed_mode_option:
        seed_mode = "date"
    elif "用户输入固定种子" in seed_mode_option:
        seed_mode = "fixed"
    else:
        seed_mode = "random"
    # ===
    def run_backtest_method_a(draws, num_bets, num_count, test_periods, train_window,
                          seed_mode, fixed_seed_value, sum_predict_method,
                          hot_count=6, cold_count=1, hot_range=(0, 10),
                          hot_temperature=0.8, cold_temperature=0.8, zone_window=15):
        """方法A回测函数"""
        if len(draws) < train_window + test_periods:
            return None
        
        method_seed_offset = 50
        total_cost = 0
        total_prize = 0
        win_count = 0
        prize_details = []
        
        from math import comb
        cost_per_bet = comb(num_count, 6) * 5
        
        for i in range(test_periods):
            train_draws = draws[:-(test_periods - i)]
            test_draw = draws[-(test_periods - i)]
            test_period = test_draw.get('period', '')
            
            if seed_mode == "date":
                test_date = test_draw.get('date')
                if test_date:
                    try:
                        dt = datetime.strptime(test_date[:10], '%Y-%m-%d')
                        seed_val = int(datetime(dt.year, dt.month, dt.day, 21, 30).timestamp())
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
            
            # 获取加分配置
            zone_bonus_config = {
                1: st.session_state.get('zone1_bonus', 8),
                2: st.session_state.get('zone2_bonus', 5),
                3: st.session_state.get('zone3_bonus', 3),
                4: st.session_state.get('zone4_bonus', 0),
                5: st.session_state.get('zone5_bonus', 0),
                6: st.session_state.get('zone6_bonus', 0),
                7: st.session_state.get('zone7_bonus', 0),
            }
            
            pattern_config = {
                "gap_2": st.session_state.get('pattern_gap2', 15),
                "gap_3": st.session_state.get('pattern_gap3', 10),
                "edge_normal": st.session_state.get('pattern_edge_normal', 8),
                "edge_special": st.session_state.get('pattern_edge_special', 6),
                "consecutive": st.session_state.get('pattern_consecutive', 5),
                "alternate": st.session_state.get('pattern_alternate', 5),
                "max": st.session_state.get('pattern_max', 25),
            }
            
            cold_config = {
                "frequency_acceleration": st.session_state.get('cold_freq_acc', 12),
                "miss_13_15": st.session_state.get('cold_miss_13_15', 8),
                "consecutive": st.session_state.get('cold_consecutive', 10),
                "cold_return": st.session_state.get('cold_return', 8),
                "cold_neighbor": st.session_state.get('cold_neighbor', 5),
                "max": st.session_state.get('cold_max', 20),
            }
            
            bets = generate_bets_method_a(
                train_draws, num_bets, num_count,
                hot_count=hot_count, cold_count=cold_count,
                hot_range=hot_range,
                hot_temperature=hot_temperature,
                cold_temperature=cold_temperature,
                zone_window=zone_window,
                zone_bonus_config=zone_bonus_config,
                pattern_config=pattern_config,
                cold_config=cold_config,
                sum_predict_method=sum_predict_method,
                random_seed=seed_val
            )
            
            best_prize = 0
            best_match_score = 0
            
            for bet in bets:
                prize = calculate_7code_prize(bet['numbers'], test_draw)
                match_score = get_best_match_score(bet['numbers'], test_draw)
                
                if prize > best_prize:
                    best_prize = prize
                    best_match_score = match_score
            
            total_cost += num_bets * cost_per_bet
            total_prize += best_prize
            
            if best_prize > 0:
                win_count += 1
                if best_prize >= 5090000:
                    prize_desc = "509万"
                elif best_prize >= 1530000:
                    prize_desc = "153万"
                elif best_prize >= 30800:
                    prize_desc = "3.08万"
                elif best_prize >= 10560:
                    prize_desc = "10,560"
                elif best_prize >= 1040:
                    prize_desc = "1,040"
                elif best_prize >= 4800:
                    prize_desc = "4,800"
                elif best_prize >= 1600:
                    prize_desc = "1,600"
                elif best_prize >= 520:
                    prize_desc = "520"
                elif best_prize >= 320:
                    prize_desc = "320"
                elif best_prize >= 160:
                    prize_desc = "160"
                elif best_prize >= 80:
                    prize_desc = "80"
                else:
                    prize_desc = str(best_prize)
                
                if best_match_score == int(best_match_score):
                    match_display = f"{int(best_match_score)}"
                else:
                    match_display = f"{best_match_score:.1f}"
                
                prize_details.append(f"{test_period}({match_display}, {prize_desc})")
        
        net = total_prize - total_cost
        roi = (net / total_cost) * 100 if total_cost > 0 else 0
        win_rate = (win_count / test_periods) * 100 if test_periods > 0 else 0
        
        return {
            "方法": "方法A: 分池评分法",
            "ROI": roi,
            "总成本": total_cost,
            "总奖金": total_prize,
            "净收益": net,
            "中奖率": win_rate,
            "中奖明细": ", ".join(prize_details) if prize_details else "无"
        }
#---------
#---------------
def run_backtest_method_b(draws, num_bets, num_count, test_periods, train_window,
                          seed_mode, fixed_seed_value, sum_predict_method) -> Optional[Dict]:
    """
    方法B回测函数（新胆拖混合，基于方法A评分）
    """
    if len(draws) < train_window + test_periods:
        return None
    
    method_seed_offset = 600  # 方法B的偏移量
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    prize_details = []
    
    from math import comb
    cost_per_bet = comb(num_count, 6) * 5
    
    for i in range(test_periods):
        train_draws = draws[:-(test_periods - i)]
        test_draw = draws[-(test_periods - i)]
        test_period = test_draw.get('period', '')
        
        # 设置随机种子
        if seed_mode == "date":
            test_date = test_draw.get('date')
            if test_date:
                try:
                    dt = datetime.strptime(test_date[:10], '%Y-%m-%d')
                    seed_val = int(datetime(dt.year, dt.month, dt.day, 21, 30).timestamp())
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
        
        # 生成投注
        bets = generate_bets_method_b(
            train_draws, num_bets, num_count, sum_predict_method, seed_val
        )
        
        best_prize = 0
        best_match_score = 0
        
        for bet in bets:
            prize = calculate_7code_prize(bet['numbers'], test_draw)
            match_score = get_best_match_score(bet['numbers'], test_draw)
            
            if prize > best_prize:
                best_prize = prize
                best_match_score = match_score
        
        total_cost += num_bets * cost_per_bet
        total_prize += best_prize
        
        if best_prize > 0:
            win_count += 1
            if best_prize >= 5090000:
                prize_desc = "509万"
            elif best_prize >= 1530000:
                prize_desc = "153万"
            elif best_prize >= 30800:
                prize_desc = "3.08万"
            elif best_prize >= 10560:
                prize_desc = "10,560"
            elif best_prize >= 1040:
                prize_desc = "1,040"
            elif best_prize >= 4800:
                prize_desc = "4,800"
            elif best_prize >= 1600:
                prize_desc = "1,600"
            elif best_prize >= 520:
                prize_desc = "520"
            elif best_prize >= 320:
                prize_desc = "320"
            elif best_prize >= 160:
                prize_desc = "160"
            elif best_prize >= 80:
                prize_desc = "80"
            else:
                prize_desc = str(best_prize)
            
            if best_match_score == int(best_match_score):
                match_display = f"{int(best_match_score)}"
            else:
                match_display = f"{best_match_score:.1f}"
            
            prize_details.append(f"{test_period}({match_display}, {prize_desc})")
    
    net = total_prize - total_cost
    roi = (net / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (win_count / test_periods) * 100 if test_periods > 0 else 0
    
    return {
        "方法": "方法B: 新胆拖混合",
        "ROI": roi,
        "总成本": total_cost,
        "总奖金": total_prize,
        "净收益": net,
        "中奖率": win_rate,
        "中奖明细": ", ".join(prize_details) if prize_details else "无"
    }

# ========== 运行回测按钮 ==========
if st.button("▶️ 运行5种方法回测", type="primary", key="run_backtest_all"):
    # 7种方法列表（根据复选框筛选）
    methods = []
    if enable_method_b:
        methods.append(("方法B: 新胆拖混合", "方法B"))
    if enable_method_a:
        methods.append(("方法A: 分池评分法", "方法A"))
    if enable_method1:
        methods.append(("方法1: 当前方法", "方法1"))
    if enable_method2:
        methods.append(("方法2: 胆拖混合", "方法2"))
    if enable_method3:
        methods.append(("方法3: LightGBM", "方法3"))
    if enable_method4:
        methods.append(("方法4: XGBoost+NN", "方法4"))
    if enable_method5:
        methods.append(("方法5: 综合模式", "方法5"))
    
    if not methods:
        st.warning("请至少选择一种回测方法")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        
        for idx, (display_name, method_key) in enumerate(methods):
            status_text.text(f"正在回测 {display_name}... ({idx+1}/{len(methods)})")
            
            # 根据方法选择对应的训练窗口
            if method_key == "方法B":
                bt_window = method_b_window
            elif method_key == "方法A":
                bt_window = method_a_window
            elif method_key in ["方法1", "方法2"]:
                bt_window = method1_window
            elif method_key == "方法3":
                bt_window = method3_window
            else:
                bt_window = method4_window
            
            # 调用回测函数
            if method_key == "方法B":
                result = run_backtest_method_b(
                    backtest_draws, test_bets, test_num_count,
                    test_periods, bt_window,
                    seed_mode, fixed_seed_value,
                    sum_predict_method
                )
            elif method_key == "方法A":
                config = get_method_a_config_from_session()
                result = run_backtest_method_a(
                    backtest_draws, test_bets, test_num_count,
                    test_periods, bt_window,
                    seed_mode, fixed_seed_value,
                    sum_predict_method,
                    hot_count=config.hot_count,
                    cold_count=config.cold_count,
                    hot_range=config.hot_range,
                    hot_temperature=config.hot_temperature,
                    cold_temperature=config.cold_temperature,
                    zone_window=config.zone_window
                )
            else:
                result = run_backtest_single_method(
                    backtest_draws, method_key, test_bets, test_num_count,
                    test_trend_window, test_periods, bt_window,
                    seed_mode, fixed_seed_value,
                    sum_predict_method
                )
            
            if result:
                all_results.append(result)
            
            progress_bar.progress((idx + 1) / len(methods))
        
        status_text.text("回测完成！")
        progress_bar.empty()
        
        # 显示结果表格
        if all_results:
            df_results = pd.DataFrame(all_results)
            
            st.dataframe(
                df_results.style.format({
                    'ROI': '{:.1f}%',
                    '总成本': '¥{:.0f}',
                    '总奖金': '¥{:.0f}',
                    '净收益': '¥{:.0f}',
                    '中奖率': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    '中奖明细': st.column_config.TextColumn('中奖明细', width='large')
                }
            )
            
            best_roi = all_results[0]['ROI']
            best_method = all_results[0]['方法']
            for r in all_results:
                if r['ROI'] > best_roi:
                    best_roi = r['ROI']
                    best_method = r['方法']
            
            st.success(f"🏆 最佳表现: {best_method} (ROI: {best_roi:.1f}%)")
            st.caption(f"📅 基于最近{test_periods}期回测，每期{test_bets}组{test_num_count}码复式")    

#---------------------------------
st.markdown("---")
st.caption("DFSS智能选号工具 v7.1")
st.caption("更新: 2026-05-13")
st.caption("修复: 数据排序 | 日期格式 | 回测完整实现")

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
    #---------------
    with st.expander("📖 七种AI算法对比"):
        st.markdown("""
        | 算法 | 核心原理 | 规律加权 |
        |------|---------|---------|
        | 🌟 方法A | 分池评分+线性归一化 | ✅ 已集成 |
        | 🆕 方法B | 新胆拖混合（基于方法A评分） | ✅ 已集成 |
        | 🟢 方法1 | 冷热码+和值 | ✅ 已集成 |
        | 🟡 方法2 | 胆拖混合 | ✅ 已集成 |
        | 🔵 方法3 | LightGBM | 规律特征 |
        | 🟣 方法4 | XGBoost+NN | 规律特征 |
        | 🔴 方法5 | 综合投票 | ✅ 已集成 |
        """)
    #-----------------------
    with st.expander("💰 奖金结构（7码复式半注）"):
        st.markdown("""
        | 7码中包含 | 总奖金 | 注数分布 |
        |-----------|--------|----------|
        | 3个正码 | **$80** | 4注中3码 |
        | 3正 + 特码 | **$500** | 1注中3码 + 3注中3+特 |
        | 4个正码 | **$1,040** | 3注中4码 + 4注中3码 |
        | 4正 + 特码 | **$10,560** | 2注中4+特 + 1注中4码 + 4注中3+特 |
        | 5个正码 | **~$61,600** | 复杂分布 |
        | 5正 + 特码 | **~$3,060,000** | 复杂分布 |
        | 6个正码 | **~$10,180,000** | 复杂分布 |
        """)
        st.caption("💡 7码复式 = 7注，每注半注$5，总成本$35")
    
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

#---------------

print("第4部分加载完成 (v7.1 - 修复版)")
print("=" * 60)
print("所有代码加载完成！六合彩AI智能选号工具 v7.1 已就绪。")
print("=" * 60)

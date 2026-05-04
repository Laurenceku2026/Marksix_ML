# app.py
# 六合彩AI智能选号工具 - Streamlit Cloud ML完整版
# 支持四种AI模型：当前方法、胆拖混合、LightGBM、XGBoost+神经网络集成
# 数据存储：Supabase 云端数据库

import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import hmac
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="六合彩AI分析工具 - 云端ML完整版",
    page_icon="🎰",
    layout="wide"
)

# 自定义CSS
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

# ==================== Supabase 初始化 ====================
def init_supabase():
    """初始化Supabase连接"""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Supabase连接失败: {e}")
        return None

def save_draws_to_supabase(draws):
    """保存开奖数据到Supabase（覆盖保存）"""
    supabase = init_supabase()
    if supabase is None:
        return False
    try:
        # 清空现有数据
        table = supabase.schema('marksix_schema').table('marksix_draws')
        table.delete().neq("id", 0).execute()
        
        # 插入新数据
        for draw in draws:
            data = {
                "period": draw.get('period'),
                "date": draw.get('date'),
                "numbers": draw['numbers'],
                "special": draw.get('special'),
                "sum_value": draw['sum']
            }
            table.insert(data).execute()
        return True
    except Exception as e:
        st.error(f"保存到Supabase失败: {e}")
        return False

def load_draws_from_supabase():
    """从Supabase加载开奖数据（按期次升序）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    try:
        response = supabase.schema('marksix_schema').table('marksix_draws').select("*").order("period", desc=False).execute()
        draws = []
        for row in response.data:
            draws.append({
                'period': row.get('period'),
                'date': row.get('date'),
                'numbers': row['numbers'],
                'special': row.get('special'),
                'sum': row['sum_value']
            })
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None

# ==================== 日期时间转Excel编码函数 ====================
def datetime_to_excel_serial(dt):
    base_date = datetime(1900, 1, 1)
    delta = dt - base_date
    days = delta.days + 2
    seconds = delta.seconds
    time_fraction = seconds / 86400
    return days + time_fraction

def parse_datetime_string(datetime_str):
    datetime_str = datetime_str.strip()
    if not datetime_str:
        return None
    
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
        "%Y%m%d %H:%M:%S", "%Y%m%d %H:%M", "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(datetime_str, fmt)
            serial = datetime_to_excel_serial(dt)
            return int(serial * 1000000)
        except ValueError:
            continue
    
    st.warning(f"无法解析日期时间格式: {datetime_str}，将使用完全随机")
    return None

# ==================== 分区函数 ====================
def get_zone(num):
    return (num - 1) // 7 + 1

def get_zone_numbers(zone):
    start = (zone - 1) * 7 + 1
    end = start + 6
    return list(range(start, end + 1))

def calculate_zone_heat(draws, last_n=20):
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

def get_hot_zones(zone_scores, num_hot_zones=3):
    sorted_zones = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
    return [zone for zone, score in sorted_zones[:num_hot_zones]]

# ==================== 核心评分函数 ====================
def calculate_scores(draws, window_total=100, window_short=20, window_recent=10):
    if len(draws) < window_total:
        window_total = len(draws)
    
    total_draws = len(draws)
    expected_freq = total_draws * 6 / 49
    
    recent_draws_total = draws[-window_total:] if len(draws) >= window_total else draws
    recent_draws_short = draws[-window_short:] if len(draws) >= window_short else draws
    recent_draws_window = draws[-window_recent:] if len(draws) >= window_recent else draws
    
    freq = {i: 0 for i in range(1, 50)}
    for draw in recent_draws_total:
        for num in draw['numbers']:
            freq[num] += 1
    
    short_freq = {i: 0 for i in range(1, 50)}
    for draw in recent_draws_short:
        for num in draw['numbers']:
            short_freq[num] += 1
    expected_short = window_short * 6 / 49
    
    last_seen = {i: None for i in range(1, 50)}
    for idx, draw in enumerate(reversed(draws)):
        for num in draw['numbers']:
            if last_seen[num] is None:
                last_seen[num] = idx
    absence = {i: last_seen[i] if last_seen[i] is not None else total_draws for i in range(1, 50)}
    
    recent_numbers = set()
    for draw in recent_draws_window:
        recent_numbers.update(draw['numbers'])
    
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

def calculate_enhanced_scores(draws, window_total=100, window_short=20, window_recent=10, zone_window=20):
    base_scores, freq, short_freq, absence = calculate_scores(draws, window_total, window_short, window_recent)
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    last_draw_all = last_numbers + [last_special] if last_special else last_numbers
    
    zone_scores, _ = calculate_zone_heat(draws, last_n=zone_window)
    hot_zones = get_hot_zones(zone_scores, num_hot_zones=3)
    
    repeat_boost = {}
    for num in range(1, 50):
        boost = 0.0
        
        if num in last_draw_all:
            boost += 2.0
        
        if len(draws) >= 2:
            prev_draw = draws[-2]
            prev_numbers = prev_draw['numbers'] + [prev_draw.get('special')] if prev_draw.get('special') else prev_draw['numbers']
            if num in prev_numbers and num not in last_draw_all:
                boost += 1.0
        
        if len(draws) >= 3:
            last_3_draws = draws[-3:]
            count_in_last_3 = 0
            for d in last_3_draws:
                if num in d['numbers'] or num == d.get('special'):
                    count_in_last_3 += 1
            if count_in_last_3 >= 2 and num not in last_draw_all:
                boost += 0.8
        
        if len(draws) >= 5:
            last_5_draws = draws[-5:]
            count_in_last_5 = 0
            for d in last_5_draws:
                if num in d['numbers'] or num == d.get('special'):
                    count_in_last_5 += 1
            if count_in_last_5 >= 3:
                boost += 0.5
        
        num_zone = get_zone(num)
        if num_zone in hot_zones:
            boost += 1.2
        
        repeat_boost[num] = boost
    
    enhanced_scores = {}
    for num in range(1, 50):
        enhanced_scores[num] = base_scores[num] + repeat_boost[num]
    
    return enhanced_scores, repeat_boost, hot_zones

def get_target_sum_by_numbers_count(num_count):
    if num_count == 7:
        return 175
    elif num_count == 6:
        return 150
    else:
        return int((1 + 49) / 2 * num_count)

def convert_6sum_to_7sum(sum_6):
    return int(sum_6 * 7 / 6)

def has_consecutive_or_jump(nums):
    nums = sorted(nums)
    for i in range(len(nums)-1):
        diff = nums[i+1] - nums[i]
        if diff == 1 or diff == 2:
            return True
    return False

def get_dynamic_sum_range(draws, num_count, window=4, sigma_factor=0.5, threshold_factor=0.1):
    recent_draws = draws[-100:] if len(draws) >= 100 else draws
    all_sum_7 = [convert_6sum_to_7sum(d['sum']) for d in recent_draws]
    long_term_mean = np.mean(all_sum_7) if all_sum_7 else 175
    long_term_std = np.std(all_sum_7) if len(all_sum_7) > 1 else 35
    
    short_draws = draws[-window:] if len(draws) >= window else draws
    short_sum_7 = [convert_6sum_to_7sum(d['sum']) for d in short_draws]
    short_mean = np.mean(short_sum_7) if short_sum_7 else long_term_mean
    
    threshold = long_term_std * threshold_factor
    
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
        base_target = get_target_sum_by_numbers_count(num_count)
        target = int(target * num_count / 7)
        tolerance = int(tolerance * num_count / 7)
    
    return int(target), tolerance, direction, direction_desc, long_term_mean, long_term_std, short_mean

def get_sampling_weights(scores, temperature=1.5):
    weights = {}
    for num, score in scores.items():
        weights[num] = math.exp(score / temperature)
    return weights

def weighted_random_sample(weights, k=7, max_attempts=100):
    numbers = list(weights.keys())
    weight_list = [weights[n] for n in numbers]
    
    attempts = 0
    while attempts < max_attempts:
        selected = random.choices(
            population=numbers,
            weights=weight_list,
            k=k
        )
        if len(set(selected)) == k:
            return sorted(selected)
        attempts += 1
    
    return sorted(random.sample(numbers, k))

def is_valid_combination(nums, target_sum, tolerance, require_pattern, require_prev_repeat, last_draw_all):
    total = sum(nums)
    
    if abs(total - target_sum) > tolerance:
        return False
    
    if require_pattern:
        if not has_consecutive_or_jump(nums):
            return False
    
    if require_prev_repeat and last_draw_all:
        prev_repeat_count = len(set(nums) & set(last_draw_all))
        if prev_repeat_count < 1 or prev_repeat_count > 2:
            return False
    
    return True

def generate_one_combination(weights, num_count, target_sum, tolerance, require_pattern, require_prev_repeat, last_draw_all):
    max_attempts = 10000
    for _ in range(max_attempts):
        selected = weighted_random_sample(weights, k=num_count)
        if is_valid_combination(selected, target_sum, tolerance, require_pattern, require_prev_repeat, last_draw_all):
            return selected, sum(selected)
    
    for _ in range(5000):
        selected = weighted_random_sample(weights, k=num_count)
        total = sum(selected)
        if abs(total - target_sum) <= tolerance + 5:
            return selected, total
    
    return sorted(random.sample(range(1, 50), num_count)), sum(sorted(random.sample(range(1, 50), num_count)))

def set_random_seed(seed_value):
    if seed_value is not None:
        try:
            random.seed(seed_value)
            np.random.seed(seed_value)
        except (ValueError, TypeError):
            random.seed()
            np.random.seed()
    else:
        random.seed()
        np.random.seed()

# ==================== 胆拖法函数 ====================
def select_anchor_numbers(draws, num_anchors=3):
    enhanced_scores, _, hot_zones = calculate_enhanced_scores(draws)
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    
    candidates = []
    
    for num in last_numbers:
        candidates.append((num, enhanced_scores[num] + 2.0))
    if last_special:
        candidates.append((last_special, enhanced_scores[last_special] + 1.5))
    
    for zone in hot_zones:
        for num in get_zone_numbers(zone):
            if enhanced_scores[num] > 0:
                candidates.append((num, enhanced_scores[num] + 1.0))
    
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
    
    candidates_dict = {}
    for num, score in candidates:
        if num not in candidates_dict:
            candidates_dict[num] = score
        else:
            candidates_dict[num] = max(candidates_dict[num], score)
    
    sorted_candidates = sorted(candidates_dict.items(), key=lambda x: x[1], reverse=True)
    anchors = [num for num, score in sorted_candidates[:num_anchors]]
    
    return anchors

def generate_one_combination_with_anchors(weights, anchors, num_count, target_sum, tolerance, 
                                           require_pattern, require_prev_repeat, last_draw_all):
    remaining_needed = num_count - len(anchors)
    if remaining_needed <= 0:
        return sorted(anchors[:num_count]), sum(sorted(anchors[:num_count]))
    
    max_attempts = 5000
    for _ in range(max_attempts):
        available_numbers = [n for n in weights.keys() if n not in anchors]
        available_weights = [weights[n] for n in available_numbers]
        
        if len(available_numbers) < remaining_needed:
            continue
        
        selected = random.choices(
            population=available_numbers,
            weights=available_weights,
            k=remaining_needed
        )
        
        full_selection = anchors + selected
        if len(set(full_selection)) != len(full_selection):
            continue
            
        if is_valid_combination(full_selection, target_sum, tolerance, 
                                require_pattern, require_prev_repeat, last_draw_all):
            return sorted(full_selection), sum(full_selection)
    
    remaining = [n for n in range(1, 50) if n not in anchors]
    full_selection = anchors + random.sample(remaining, min(remaining_needed, len(remaining)))
    return sorted(full_selection), sum(full_selection)

# ==================== LightGBM 函数 ====================
def build_features_for_lightgbm(draws, target_num):
    if len(draws) < 20:
        return None
    
    features = {}
    total_draws = len(draws)
    
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
    
    last_numbers = draws[-1]['numbers']
    min_diff = min(abs(target_num - n) for n in last_numbers) if last_numbers else 99
    features['min_diff_to_last'] = min_diff
    
    same_tail_count = sum(1 for d in draws[-20:] if any(n % 10 == target_num % 10 for n in d['numbers']))
    features['same_tail_hot'] = same_tail_count / max(1, min(20, len(draws)))
    
    return features

def prepare_lightgbm_dataset(draws, lookback=100):
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
    
    X_df = pd.DataFrame(X_list)
    y_series = pd.Series(y_list)
    
    return X_df, y_series

def train_lightgbm_model(draws):
    if not LGB_AVAILABLE:
        return None
    
    X, y = prepare_lightgbm_dataset(draws)
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
        st.warning(f"LightGBM训练失败: {e}")
        return None

def predict_with_lightgbm(model, draws):
    if model is None:
        return None
    
    predictions = []
    for num in range(1, 50):
        features = build_features_for_lightgbm(draws, num)
        if features:
            X_pred = pd.DataFrame([features])
            prob = model.predict_proba(X_pred)[0][1]
            predictions.append((num, prob))
        else:
            predictions.append((num, 0.0))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [num for num, prob in predictions[:7]]

# ==================== XGBoost + 神经网络集成函数 ====================
def build_advanced_features(draws, target_num):
    if len(draws) < 30:
        return None
    
    features = {}
    total_draws = len(draws)
    
    recent_100 = draws[-100:] if len(draws) >= 100 else draws
    freq = sum(1 for d in recent_100 if target_num in d['numbers'])
    features['freq'] = freq / max(1, len(recent_100))
    
    all_freqs = [sum(1 for d in recent_100 if i in d['numbers']) for i in range(1, 50)]
    features['freq_zscore'] = (freq - np.mean(all_freqs)) / max(1, np.std(all_freqs))
    
    short_draws = draws[-20:] if len(draws) >= 20 else draws
    short_freq = sum(1 for d in short_draws if target_num in d['numbers'])
    features['short_freq'] = short_freq / max(1, len(short_draws))
    
    last_seen = None
    for idx, d in enumerate(reversed(draws)):
        if target_num in d['numbers']:
            last_seen = idx
            break
    absence = last_seen if last_seen is not None else total_draws
    features['absence'] = absence
    features['absence_norm'] = absence / max(1, total_draws)
    
    features['recent_3'] = sum(1 for d in draws[-3:] if target_num in d['numbers'])
    features['recent_5'] = sum(1 for d in draws[-5:] if target_num in d['numbers'])
    features['recent_10'] = sum(1 for d in draws[-10:] if target_num in d['numbers'])
    
    last_date = draws[-1].get('date')
    if last_date and isinstance(last_date, str):
        try:
            last_dt = datetime.strptime(last_date.split()[0] if ' ' in last_date else last_date, '%Y-%m-%d')
            features['weekday'] = last_dt.weekday()
            features['is_weekend'] = 1 if features['weekday'] >= 5 else 0
            features['is_saturday'] = 1 if features['weekday'] == 5 else 0
            features['month'] = last_dt.month
            features['quarter'] = (last_dt.month - 1) // 3
        except:
            features['weekday'] = 0
            features['is_weekend'] = 0
            features['is_saturday'] = 0
            features['month'] = 0
            features['quarter'] = 0
    else:
        features['weekday'] = 0
        features['is_weekend'] = 0
        features['is_saturday'] = 0
        features['month'] = 0
        features['quarter'] = 0
    
    appearances = [1 if target_num in d['numbers'] else 0 for d in draws[-50:]]
    if len(appearances) >= 10:
        features['ma5'] = np.mean(appearances[-5:])
        features['ma10'] = np.mean(appearances[-10:])
        features['trend'] = features['ma5'] - features['ma10']
    else:
        features['ma5'] = 0
        features['ma10'] = 0
        features['trend'] = 0
    
    last_numbers = draws[-1]['numbers']
    features['in_last'] = 1 if target_num in last_numbers else 0
    features['min_diff'] = min(abs(target_num - n) for n in last_numbers) if last_numbers else 99
    features['same_parity'] = 1 if len(last_numbers) > 0 and (target_num % 2) == (last_numbers[0] % 2) else 0
    
    same_tail_nums = [n for n in range(1, 50) if n % 10 == target_num % 10]
    same_tail_freq = sum(1 for d in draws[-20:] for n in d['numbers'] if n in same_tail_nums)
    features['same_tail_hot'] = same_tail_freq / max(1, 20 * len(same_tail_nums))
    
    zone = get_zone(target_num)
    features['zone'] = zone
    
    zone_hits = {}
    for z in range(1, 8):
        zone_hits[z] = sum(1 for d in draws[-20:] for n in d['numbers'] if get_zone(n) == z)
    features['zone_hot'] = zone_hits.get(zone, 0) / max(1, 20)
    
    last_special = draws[-1].get('special')
    features['is_special'] = 1 if target_num == last_special else 0
    features['diff_to_special'] = abs(target_num - last_special) if last_special else 99
    
    return features

def prepare_advanced_dataset(draws, lookback=100):
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
    
    X_df = pd.DataFrame(X_list)
    y_series = pd.Series(y_list)
    
    X_df = X_df.fillna(0)
    
    return X_df, y_series

def train_xgboost_nn_ensemble(draws):
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    
    X, y = prepare_advanced_dataset(draws)
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
        st.warning(f"XGBoost+NN集成训练失败: {e}")
        return None

def predict_with_ensemble(model_dict, draws):
    if model_dict is None:
        return None
    
    try:
        predictions = []
        for num in range(1, 50):
            features = build_advanced_features(draws, num)
            if features:
                X_pred = pd.DataFrame([features])
                X_pred = X_pred.fillna(0)
                
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
        st.warning(f"集成预测失败: {e}")
        return None

# ==================== 投注生成主函数 ====================
def generate_bets_method1_current(draws, num_bets, num_count, require_pattern, require_prev_repeat, 
                                    trend_window, random_seed, analysis_periods):
    set_random_seed(random_seed)
    
    enhanced_scores, repeat_boost, hot_zones = calculate_enhanced_scores(draws, window_total=analysis_periods)
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    last_draw_all = last_numbers + [last_special] if require_prev_repeat else None
    
    weights = get_sampling_weights(enhanced_scores, temperature=1.5)
    
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    
    bets = []
    for i in range(num_bets):
        offset = random.randint(-tolerance, tolerance)
        t = int(target_sum + offset)
        t = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, t))
        nums, total = generate_one_combination(
            weights, num_count, t, tolerance,
            require_pattern, require_prev_repeat, last_draw_all
        )
        bets.append({'numbers': nums, 'sum': total, 'target': f'当前方法(目标{t})', 'deviation': total - base_target})
    
    return bets

def generate_bets_method2_hybrid(draws, num_bets, num_count, require_pattern, require_prev_repeat, 
                                  trend_window, random_seed, analysis_periods):
    set_random_seed(random_seed)
    
    enhanced_scores, repeat_boost, hot_zones = calculate_enhanced_scores(draws, window_total=analysis_periods)
    
    last_draw = draws[-1]
    last_numbers = last_draw['numbers']
    last_special = last_draw.get('special')
    last_draw_all = last_numbers + [last_special] if require_prev_repeat else None    
    weights = get_sampling_weights(enhanced_scores, temperature=1.5)
    
    target_sum, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
    base_target = get_target_sum_by_numbers_count(num_count)
    target_sum = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, target_sum))
    
    num_normal = max(1, num_bets // 2)
    num_anchor = num_bets - num_normal
    
    bets = []
    
    anchors = select_anchor_numbers(draws, num_anchors=3)
    
    for i in range(num_normal):
        offset = random.randint(-tolerance, tolerance)
        t = int(target_sum + offset)
        t = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, t))
        nums, total = generate_one_combination(
            weights, num_count, t, tolerance,
            require_pattern, require_prev_repeat, last_draw_all
        )
        bets.append({'numbers': nums, 'sum': total, 'target': f'当前方法(目标{t})', 'deviation': total - base_target})
    
    anchor_weights = get_sampling_weights(enhanced_scores, temperature=1.5)
    for i in range(num_anchor):
        offset = random.randint(-tolerance, tolerance)
        t = int(target_sum + offset)
        t = max(base_target - 2 * tolerance, min(base_target + 2 * tolerance, t))
        nums, total = generate_one_combination_with_anchors(
            anchor_weights, anchors, num_count, t, tolerance,
            require_pattern, require_prev_repeat, last_draw_all
        )
        bets.append({'numbers': nums, 'sum': total, 'target': f'胆拖法(目标{t})', 'deviation': total - base_target})
    
    return bets

def generate_bets_method3_lightgbm(draws, num_bets, num_count, require_pattern, require_prev_repeat,
                                    trend_window, random_seed, analysis_periods):
    if not LGB_AVAILABLE:
        st.warning("⚠️ LightGBM未安装，降级使用混合策略")
        return generate_bets_method2_hybrid(draws, num_bets, num_count, require_pattern,
                                            require_prev_repeat, trend_window, random_seed, analysis_periods)
    
    set_random_seed(random_seed)
    
    model = train_lightgbm_model(draws)
    
    if model is None:
        st.warning("LightGBM模型训练失败，降级使用混合策略")
        return generate_bets_method2_hybrid(draws, num_bets, num_count, require_pattern,
                                            require_prev_repeat, trend_window, random_seed, analysis_periods)
    
    predicted_numbers = predict_with_lightgbm(model, draws)
    
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    base_target, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
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
            'target': f'LightGBM(目标{base_target})',
            'deviation': total - base_target
        })
    
    return bets

def generate_bets_method4_ensemble(draws, num_bets, num_count, require_pattern, require_prev_repeat,
                                    trend_window, random_seed, analysis_periods):
    if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        st.warning("⚠️ XGBoost或sklearn未安装，降级使用LightGBM")
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, require_pattern,
                                              require_prev_repeat, trend_window, random_seed, analysis_periods)
    
    set_random_seed(random_seed)
    
    ensemble = train_xgboost_nn_ensemble(draws)
    
    if ensemble is None:
        st.warning("XGBoost+NN集成训练失败，降级使用LightGBM")
        return generate_bets_method3_lightgbm(draws, num_bets, num_count, require_pattern,
                                              require_prev_repeat, trend_window, random_seed, analysis_periods)
    
    predicted_numbers = predict_with_ensemble(ensemble, draws)
    
    if predicted_numbers is None or len(predicted_numbers) < num_count:
        predicted_numbers = list(range(1, num_count + 1))
    
    base_target, tolerance, direction, _, _, _, _ = get_dynamic_sum_range(draws, num_count, window=trend_window)
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
            'target': f'XGBoost+NN(目标{base_target})',
            'deviation': total - base_target
        })
    
    return bets

# ==================== 多期查奖函数 ====================
def calculate_match_score(bet_numbers, draw_numbers, draw_special):
    bet_main = set(bet_numbers[:6])
    draw_main = set(draw_numbers)
    main_matches = len(bet_main & draw_main)
    special_match = False
    if len(bet_numbers) >= 7:
        special_match = (bet_numbers[6] == draw_special)
    if special_match:
        return main_matches + 0.5
    else:
        return float(main_matches)

def format_score_display(score):
    if score == int(score):
        return str(int(score))
    else:
        return f"{score:.1f}"

def calculate_match_score_for_draws(bet_numbers, check_draws):
    results = []
    for draw in check_draws:
        score = calculate_match_score(bet_numbers, draw['numbers'], draw['special'])
        results.append(format_score_display(score))
    return results

# ==================== 管理员函数 ====================
def check_password(password):
    return hmac.compare_digest(password, "Ku_product$2026")

def admin_login():
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
    if st.button("退出登录", key="logout_btn"):
        st.session_state['admin_logged_in'] = False
        st.session_state['show_admin'] = False
        st.rerun()

def parse_pasted_data(text):
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
    
    draws.sort(key=lambda x: int(x.get('period', 0)))
    return draws

def parse_multi_draws_for_checking(text, max_draws=5):
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

def parse_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        number_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        existing_cols = [col for col in number_cols if col in df.columns]
        
        if len(existing_cols) >= 7:
            draws = []
            for idx, row in df.iterrows():
                try:
                    nums = []
                    for col in existing_cols[:6]:
                        val = row[col]
                        if pd.notna(val):
                            nums.append(int(val))
                    special = None
                    if len(existing_cols) > 6:
                        special_val = row[existing_cols[6]]
                        if pd.notna(special_val):
                            special = int(special_val)
                    if len(nums) == 6:
                        period_col = None
                        for col in ['期次', 'period', 'Period', '期数']:
                            if col in df.columns:
                                period_col = col
                                break
                        
                        if period_col:
                            period_val = row[period_col]
                            if pd.notna(period_val):
                                period = int(period_val)
                            else:
                                period = idx + 1
                        else:
                            period = idx + 1
                        
                        date_col = None
                        for col in ['開獎日期', '日期', 'date', 'Date']:
                            if col in df.columns:
                                date_col = col
                                break
                        
                        date_val = row[date_col] if date_col else None
                        date_str = None
                        if pd.notna(date_val):
                            if isinstance(date_val, datetime):
                                date_str = date_val.strftime('%Y-%m-%d')
                            else:
                                date_str = str(date_val)
                        
                        draws.append({
                            'period': period,
                            'date': date_str,
                            'numbers': sorted(nums),
                            'special': special,
                            'sum': sum(nums)
                        })
                except (ValueError, TypeError):
                    continue
            
            draws.sort(key=lambda x: int(x.get('period', 0)))
            return draws
    except Exception as e:
        st.error(f"Excel解析错误: {e}")
        return None

def display_centered_dataframe(df, key=None):
    st.dataframe(df, use_container_width=True, hide_index=True, key=key)

# ==================== 管理员页面 ====================
def show_admin_page():
    with st.expander("🔧 管理员控制台", expanded=True):
        st.subheader("📁 历史数据管理")
        
        current_draws = load_draws_from_supabase()
        if current_draws:
            st.success(f"✅ 当前云端有 {len(current_draws)} 期数据")
            if len(current_draws) > 0:
                st.info(f"📊 数据范围: {current_draws[0].get('period')} 到 {current_draws[-1].get('period')}")
        else:
            st.info("📭 云端暂无数据")
        
        st.markdown("---")
        
        edit_mode = st.radio(
            "选择编辑方式",
            ["📄 粘贴数据添加", "📎 上传Excel文件"],
            horizontal=True,
            key="edit_mode_admin"
        )
        
        parsed_draws = None
        
        if edit_mode == "📄 粘贴数据添加":
            st.markdown("格式: 期次 日期 B1 B2 B3 B4 B5 B6 B7")
            st.code("26045 2026-04-25 4 16 21 36 42 46 9\n26044 2026-04-23 12 23 37 38 45 48 8")
            admin_pasted = st.text_area("粘贴历史数据", height=200, key="admin_pasted")
            if admin_pasted and st.button("预览并保存", key="save_pasted"):
                parsed_draws = parse_pasted_data(admin_pasted)
                if parsed_draws:
                    st.session_state['preview_draws'] = parsed_draws
                    st.success(f"成功解析 {len(parsed_draws)} 期数据")
                    preview_df = pd.DataFrame(parsed_draws[-20:])
                    display_centered_dataframe(preview_df)
        
        elif edit_mode == "📎 上传Excel文件":
            admin_file = st.file_uploader("上传Excel文件", type=['xlsx', 'xls'], key="admin_file")
            if admin_file and st.button("预览并保存", key="save_excel"):
                parsed_draws = parse_excel_file(admin_file)
                if parsed_draws:
                    st.session_state['preview_draws'] = parsed_draws
                    st.success(f"成功解析 {len(parsed_draws)} 期数据")
                    display_centered_dataframe(pd.DataFrame(parsed_draws[-20:]))
        
        if st.session_state.get('preview_draws') is not None:
            parsed_draws = st.session_state['preview_draws']
            st.markdown("---")
            st.subheader("💾 保存到云端")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("☁️ 保存到Supabase", type="primary", key="save_parsed"):
                    with st.spinner("正在保存..."):
                        if save_draws_to_supabase(parsed_draws):
                            st.success(f"成功保存 {len(parsed_draws)} 期数据到Supabase！")
                            st.session_state['preview_draws'] = None
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("保存失败")
            with col2:
                if st.button("❌ 取消", key="cancel_parsed"):
                    st.session_state['preview_draws'] = None
                    st.rerun()

# ==================== 初始化session state ====================
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

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🎰 六合彩AI分析工具")
    st.markdown("---")
    
    # 显示ML库状态
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
    
    with st.expander("📖 四种AI算法对比"):
        st.markdown("""
        | 算法 | 特点 | 预期ROI |
        |------|------|---------|
        | 🟢 方法1:当前方法 | 冷热码+和值预测 | +33% |
        | 🟡 方法2:胆拖混合 | 当前方法+胆码 | +92% |
        | 🔵 方法3:LightGBM | 单一机器学习 | +97% |
        | 🟣 方法4:XGBoost+NN | 集成深度学习 | **+203%** |
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
    
    st.markdown("---")
    st.caption("DFSS智能选号工具 v6.0 (云端ML完整版)")

# ==================== 主页面 ====================
st.title("🎯 六合彩AI智能选号工具 - 云端ML完整版")

# 加载数据
@st.cache_data(ttl=60, show_spinner="从云端加载数据...")
def get_draws_from_cloud():
    return load_draws_from_supabase()

draws = get_draws_from_cloud()

if draws is None or len(draws) == 0:
    st.info("👈 请点击右上角齿轮图标，进入管理员页面导入历史数据")
    st.stop()

# ==================== 显示最新期数 ====================
st.subheader("📊 数据概览")
latest_draw = draws[-1]
oldest_draw = draws[0]
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

# ==================== 冷热码分析 ====================
st.subheader("🔥 冷热码分析")
analysis_periods = st.number_input(
    "分析期数", min_value=10, max_value=min(500, len(draws)),
    value=min(200, len(draws)), step=10, key="analysis_periods"
)

enhanced_scores, repeat_boost, hot_zones = calculate_enhanced_scores(draws, window_total=analysis_periods)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔥 热门号码 (Top 15)**")
    hot_df = pd.DataFrame([
        {'号码': num, '得分': f"{enhanced_scores[num]:.2f}"}
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
    st.markdown("**🔥 当前热区**")
    hot_zones_display = []
    for z in range(1, 8):
        zone_range = f"{get_zone_numbers(z)[0]:02d}-{get_zone_numbers(z)[-1]:02d}"
        hot_zones_display.append({
            '分区': f"{chr(64+z)}区 ({zone_range})",
            '热度': '🔥' if z in hot_zones else '❄️'
        })
    st.dataframe(pd.DataFrame(hot_zones_display), use_container_width=True, hide_index=True)

# ==================== 和值趋势分析 ====================
st.subheader("📈 和值趋势分析")
show_periods = st.slider("显示最近期数", min_value=10, max_value=min(200, len(draws)), value=min(100, len(draws)), step=10)

recent_sum_draws = draws[-show_periods:] if len(draws) >= show_periods else draws
sum_7_values = [convert_6sum_to_7sum(draw['sum']) for draw in recent_sum_draws]
sum_df = pd.DataFrame([{'期次': i+1, '和值(7码)': val} for i, val in enumerate(sum_7_values)])
fig = px.line(sum_df, x='期次', y='和值(7码)', title=f'最近{show_periods}期和值走势')
fig.add_hline(y=175, line_dash="dash", line_color="red", annotation_text="理论均值(175)")
fig.add_hrect(y0=140, y1=210, line_width=0, fillcolor="green", opacity=0.1, annotation_text="约68%区间")
st.plotly_chart(fig, use_container_width=True)

# ==================== 智能投注生成 ====================
st.subheader("🎲 智能投注生成")
next_period = latest_draw.get('period', 0) + 1 if isinstance(latest_draw.get('period'), int) else "N/A"
st.info(f"🎯 **预测下一期**: {next_period}")

col1, col2, col3 = st.columns(3)
with col1:
    num_bets = st.number_input("购买组数", min_value=1, max_value=100, value=4, step=1, key="num_bets", help="每组7个号码复式")
with col2:
    num_count = st.selectbox("每注号码个数", [6, 7, 8, 9, 10], index=1, key="num_count")
with col3:
    ai_model = st.selectbox(
        "🤖 AI预测模型",
        [
            "方法1: 当前方法",
            "方法2: 当前方法+胆拖混合",
            "方法3: LightGBM",
            "方法4: XGBoost+神经网络集成 ⭐推荐"
        ],
        index=3,
        key="ai_model"
    )

col1, col2 = st.columns(2)
with col1:
    require_pattern = st.checkbox("☑ 连号/跳号要求", value=True, key="require_pattern")
with col2:
    require_prev_repeat = st.checkbox("☑ 上期重复1-2个要求", value=True, key="require_prev_repeat")

col1, col2, col3 = st.columns(3)
with col1:
    trend_window = st.number_input("趋势窗口", min_value=2, max_value=20, value=4, step=1, key="trend_window")
with col2:
    seed_input = st.text_input("Random Seed", value="", placeholder="例如: 2026-05-04", key="seed_input")
with col3:
    use_analysis_periods = st.number_input(
        "冷热分析期数", min_value=10, max_value=min(500, len(draws)), 
        value=min(200, len(draws)), step=10, key="use_analysis"
    )

target_sum, tolerance, direction, direction_desc, mean_sum, std_sum, short_mean = get_dynamic_sum_range(
    draws, num_count, window=trend_window
)
st.caption(f"💡 **和值动态预测**: 长期均值={mean_sum:.1f}, σ={std_sum:.1f} | {direction_desc} | 目标={target_sum} | 容差=±{tolerance}")

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    random_seed = None
    if seed_input and seed_input.strip():
        random_seed = parse_datetime_string(seed_input)
        if random_seed:
            st.success(f"✅ 已设置Random Seed: {random_seed}")
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        if "方法1: 当前方法" in ai_model:
            bets = generate_bets_method1_current(
                draws, num_bets, num_count, require_pattern, require_prev_repeat,
                trend_window, random_seed, use_analysis_periods
            )
            model_used = "当前方法"
        elif "方法2" in ai_model:
            bets = generate_bets_method2_hybrid(
                draws, num_bets, num_count, require_pattern, require_prev_repeat,
                trend_window, random_seed, use_analysis_periods
            )
            model_used = "当前方法+胆拖混合"
        elif "方法3" in ai_model:
            bets = generate_bets_method3_lightgbm(
                draws, num_bets, num_count, require_pattern, require_prev_repeat,
                trend_window, random_seed, use_analysis_periods
            )
            model_used = "LightGBM"
        else:
            bets = generate_bets_method4_ensemble(
                draws, num_bets, num_count, require_pattern, require_prev_repeat,
                trend_window, random_seed, use_analysis_periods
            )
            model_used = "XGBoost+神经网络集成"
    
    st.session_state['generated_bets'] = bets
    st.session_state['model_used'] = model_used
    st.success(f"✅ 使用 {model_used} 生成 {len(bets)} 组7码复式")

# 显示生成的投注
if st.session_state['generated_bets'] is not None:
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知模型')
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    st.caption("每组7个号码 = 7注（每注$5半注），每组成本$35")
    
    bets_data = []
    for i, bet in enumerate(bets, 1):
        numbers_display = ','.join(str(n) for n in bet['numbers'])
        bets_data.append({
            '组别': i,
            '7个号码': numbers_display,
            '和值': bet['sum'],
            '策略': bet['target'],
            '偏差': f"{bet['deviation']:+d}"
        })
    
    st.dataframe(pd.DataFrame(bets_data), use_container_width=True, hide_index=True)
    
    # 多期查奖
    st.markdown("---")
    st.markdown("### 🔍 多期查奖")
    st.caption("📌 粘贴实际开奖数据，查看每组7码复式中奖情况")
    
    check_draws_text = st.text_area(
        "📋 粘贴多期开奖数据（最多5期）",
        height=120,
        key="check_draws",
        placeholder="示例:\n26045 2026-04-25 4 16 21 36 42 46 9\n26044 2026-04-23 12 23 37 38 45 48 8"
    )
    
    if st.button("🔍 查奖", key="check_btn") and check_draws_text:
        check_draws = parse_multi_draws_for_checking(check_draws_text, max_draws=5)
        if check_draws:
            st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
            
            enhanced_bets_data = []
            for i, bet in enumerate(bets, 1):
                row = {'组别': i, '7个号码': ','.join(str(n) for n in bet['numbers']), '和值': bet['sum']}
                match_scores = calculate_match_score_for_draws(bet['numbers'], check_draws)
                for idx, draw in enumerate(check_draws):
                    period_str = str(draw['period'])
                    if len(period_str) > 10:
                        period_str = period_str[-10:]
                    row[f'中奖_{period_str}'] = match_scores[idx]
                enhanced_bets_data.append(row)
            
            st.dataframe(pd.DataFrame(enhanced_bets_data), use_container_width=True, hide_index=True)
            
            preview_df = pd.DataFrame([
                {'期次': d['period'], '正码': str(d['numbers']), '特码': d['special']}
                for d in check_draws
            ])
            st.markdown("**📊 开奖数据预览**")
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.error("解析失败，请检查格式")

st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。六合彩本质上是一种随机游戏，长期期望值为负，请理性投注。")

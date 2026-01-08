"""
Advanced Medical Imaging Metrics for Segmentation Evaluation

Клинически значимые метрики для оценки сегментации медицинских изображений:

Основные метрики:
1. Overlap Metrics: Dice, IoU, Jaccard
2. Distance Metrics: Hausdorff, Average Surface Distance
3. Volume Metrics: Volume error, Absolute volume difference
4. Statistical Metrics: MAE, MSE, RMSE
5. Clinical Metrics: Specificity, Sensitivity, PPV, NPV
6. Boundary Metrics: Boundary F1, Boundary IoU

Все метрики оптимизированы для 3D медицинских изображений.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import traceback
import time
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

# Типы для аннотаций
ArrayLike = Union[np.ndarray, List, Tuple]


class MetricCategory(Enum):
    """Категории метрик"""
    OVERLAP = "overlap"
    DISTANCE = "distance"
    VOLUME = "volume"
    STATISTICAL = "statistical"
    CLINICAL = "clinical"
    BOUNDARY = "boundary"
    QUALITY = "quality"


class MaskType(Enum):
    """Типы масок"""
    BINARY = "binary"
    PROBABILITY = "probability"
    MULTICLASS = "multiclass"
    CONTINUOUS = "continuous"


@dataclass
class MetricConfig:
    """Конфигурация для расчета метрик"""
    # Общие параметры
    epsilon: float = 1e-7
    is_binary: bool = True
    threshold: float = 0.5
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # в мм
    
    # Параметры для Hausdorff distance
    hausdorff_percentile: float = 95.0  # процентиль для HD95
    hausdorff_max_distance: float = 100.0  # максимальное расстояние в мм
    
    # Параметры для boundary metrics
    boundary_tolerance: int = 2  # толерантность в вокселях
    surface_dilation: int = 1
    
    # Клинические параметры
    min_liver_volume_ml: float = 800.0  # минимальный объем печени
    max_liver_volume_ml: float = 2500.0  # максимальный объем печени
    
    # Оптимизация
    use_multiprocessing: bool = False
    cache_results: bool = True


class SegmentationMetrics:
    """
    Расширенный класс для расчета метрик сегментации медицинских изображений
    """
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self._cache = {}
        self._execution_times = {}
        
    def _ensure_binary(self, arr: ArrayLike) -> np.ndarray:
        """Преобразование в бинарную маску"""
        arr = np.asarray(arr, dtype=np.float32)
        if self.config.is_binary:
            return (arr > self.config.threshold).astype(np.uint8)
        return arr
    
    def _validate_inputs(self, ground_truth: ArrayLike, prediction: ArrayLike) -> None:
        """Валидация входных данных"""
        gt = np.asarray(ground_truth)
        pred = np.asarray(prediction)
        
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} != Pred {pred.shape}")
        
        if gt.size == 0:
            raise ValueError("Empty ground truth array")
        
        if not np.any(gt > 0) and not np.any(pred > 0):
            warnings.warn("Both ground truth and prediction are empty masks")
    
    def _timed_execution(self, func: Callable, *args, **kwargs) -> Tuple[any, float]:
        """Измерение времени выполнения функции"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        func_name = func.__name__
        self._execution_times[func_name] = self._execution_times.get(func_name, []) + [execution_time]
        
        return result, execution_time
    
    # ===========================================================================
    # 1. OVERLAP METRICS (Метрики перекрытия)
    # ===========================================================================
    
    def dice_coefficient(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Коэффициент Соренсена-Дайса (Dice)
        
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Dice коэффициент [0, 1]
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth)
        pred = self._ensure_binary(prediction)
        
        intersection = np.sum(gt * pred)
        gt_sum = np.sum(gt)
        pred_sum = np.sum(pred)
        
        dice = (2.0 * intersection + self.config.epsilon) / (gt_sum + pred_sum + self.config.epsilon)
        return float(dice)
    
    def jaccard_index(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Индекс Жаккара (IoU)
        
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            IoU значение [0, 1]
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth)
        pred = self._ensure_binary(prediction)
        
        intersection = np.sum(gt * pred)
        union = np.sum(gt) + np.sum(pred) - intersection
        
        iou = (intersection + self.config.epsilon) / (union + self.config.epsilon)
        return float(iou)
    
    def volume_overlap_error(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Ошибка перекрытия объемов
        
        VOE = 1 - (|A ∩ B| / |A ∪ B|) = 1 - IoU
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Ошибка перекрытия объемов [0, 1]
        """
        iou = self.jaccard_index(ground_truth, prediction)
        return 1.0 - iou
    
    # ===========================================================================
    # 2. STATISTICAL METRICS (Статистические метрики)
    # ===========================================================================
    
    def mean_absolute_error(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Средняя абсолютная ошибка (MAE)
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Средняя абсолютная ошибка [0, ∞)
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth).astype(np.float32)
        pred = self._ensure_binary(prediction).astype(np.float32)
        
        mae = np.mean(np.abs(gt - pred))
        return float(mae)
    
    def mean_squared_error(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Среднеквадратическая ошибка (MSE)
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Среднеквадратическая ошибка [0, ∞)
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth).astype(np.float32)
        pred = self._ensure_binary(prediction).astype(np.float32)
        
        mse = np.mean((gt - pred) ** 2)
        return float(mse)
    
    def root_mean_squared_error(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Корень из среднеквадратической ошибки (RMSE)
        
        RMSE = √MSE
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            RMSE значение [0, ∞)
        """
        mse = self.mean_squared_error(ground_truth, prediction)
        return float(np.sqrt(mse))
    
    def normalized_mae(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Нормализованная средняя абсолютная ошибка
        
        NMAE = MAE / (max(y_true) - min(y_true))
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Нормализованная MAE [0, 1]
        """
        mae = self.mean_absolute_error(ground_truth, prediction)
        gt = self._ensure_binary(ground_truth)
        
        value_range = np.max(gt) - np.min(gt)
        if value_range == 0:
            return 0.0
        
        nmae = mae / value_range
        return float(nmae)
    
    # ===========================================================================
    # 3. DISTANCE METRICS (Метрики расстояния)
    # ===========================================================================
    
    def hausdorff_distance(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Расстояние Хаусдорфа
        
        HD = max(h(A,B), h(B,A)), где
        h(A,B) = max_{a∈A} min_{b∈B} ||a - b||
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Расстояние Хаусдорфа в мм
        """
        try:
            self._validate_inputs(ground_truth, prediction)
            
            gt = self._ensure_binary(ground_truth)
            pred = self._ensure_binary(prediction)
            
            # Получаем координаты граничных вокселей
            gt_points = self._get_surface_points(gt)
            pred_points = self._get_surface_points(pred)
            
            if len(gt_points) == 0 or len(pred_points) == 0:
                return float('inf')
            
            # Преобразуем воксели в мм
            gt_points_mm = gt_points * np.array(self.config.spacing)
            pred_points_mm = pred_points * np.array(self.config.spacing)
            
            # Вычисляем направленные расстояния Хаусдорфа
            hd1 = directed_hausdorff(gt_points_mm, pred_points_mm)[0]
            hd2 = directed_hausdorff(pred_points_mm, gt_points_mm)[0]
            
            hausdorff = max(hd1, hd2)
            
            # Ограничиваем максимальным значением
            if hausdorff > self.config.hausdorff_max_distance:
                hausdorff = self.config.hausdorff_max_distance
            
            return float(hausdorff)
            
        except Exception as e:
            warnings.warn(f"Hausdorff calculation failed: {e}")
            return float('inf')
    
    def hausdorff_distance_95(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        95-й процентиль расстояния Хаусдорфа (HD95)
        
        Более устойчивая метрика, менее чувствительная к выбросам.
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            HD95 в мм
        """
        try:
            self._validate_inputs(ground_truth, prediction)
            
            gt = self._ensure_binary(ground_truth)
            pred = self._ensure_binary(prediction)
            
            gt_points = self._get_surface_points(gt)
            pred_points = self._get_surface_points(pred)
            
            if len(gt_points) == 0 or len(pred_points) == 0:
                return float('inf')
            
            # Преобразуем в мм
            gt_points_mm = gt_points * np.array(self.config.spacing)
            pred_points_mm = pred_points * np.array(self.config.spacing)
            
            # Вычисляем все попарные расстояния
            from scipy.spatial import cKDTree
            
            tree_pred = cKDTree(pred_points_mm)
            distances_gt_to_pred, _ = tree_pred.query(gt_points_mm)
            
            tree_gt = cKDTree(gt_points_mm)
            distances_pred_to_gt, _ = tree_gt.query(pred_points_mm)
            
            # Объединяем расстояния
            all_distances = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])
            
            # Вычисляем 95-й процентиль
            hd95 = np.percentile(all_distances, self.config.hausdorff_percentile)
            
            if hd95 > self.config.hausdorff_max_distance:
                hd95 = self.config.hausdorff_max_distance
            
            return float(hd95)
            
        except Exception as e:
            warnings.warn(f"HD95 calculation failed: {e}")
            return float('inf')
    
    def average_surface_distance(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Среднее расстояние между поверхностями (ASD)
        
        ASD = (1/(|S_A|+|S_B|)) * (Σ_{a∈S_A} d(a,S_B) + Σ_{b∈S_B} d(b,S_A))
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Среднее расстояние в мм
        """
        try:
            self._validate_inputs(ground_truth, prediction)
            
            gt = self._ensure_binary(ground_truth)
            pred = self._ensure_binary(prediction)
            
            gt_points = self._get_surface_points(gt)
            pred_points = self._get_surface_points(pred)
            
            if len(gt_points) == 0 or len(pred_points) == 0:
                return float('inf')
            
            gt_points_mm = gt_points * np.array(self.config.spacing)
            pred_points_mm = pred_points * np.array(self.config.spacing)
            
            from scipy.spatial import cKDTree
            
            # Расстояния от GT к Prediction
            tree_pred = cKDTree(pred_points_mm)
            distances_gt_to_pred, _ = tree_pred.query(gt_points_mm)
            
            # Расстояния от Prediction к GT
            tree_gt = cKDTree(gt_points_mm)
            distances_pred_to_gt, _ = tree_gt.query(pred_points_mm)
            
            # Среднее расстояние
            asd = (np.sum(distances_gt_to_pred) + np.sum(distances_pred_to_gt)) / \
                  (len(distances_gt_to_pred) + len(distances_pred_to_gt))
            
            return float(asd)
            
        except Exception as e:
            warnings.warn(f"ASD calculation failed: {e}")
            return float('inf')
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Получение точек поверхности маски"""
        # Используем морфологические операции для выделения границ
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure=structure)
        boundaries = mask ^ eroded
        
        # Получаем координаты граничных вокселей
        surface_points = np.argwhere(boundaries)
        
        # Если нет граничных точек, используем все ненулевые точки
        if len(surface_points) == 0:
            surface_points = np.argwhere(mask)
        
        return surface_points
    
    # ===========================================================================
    # 4. VOLUME METRICS (Объемные метрики)
    # ===========================================================================
    
    def volume_metrics(self, ground_truth: ArrayLike, prediction: ArrayLike) -> Dict[str, float]:
        """
        Объемные метрики
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Словарь с объемными метриками
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth)
        pred = self._ensure_binary(prediction)
        
        voxel_volume_mm3 = np.prod(self.config.spacing)
        voxel_volume_ml = voxel_volume_mm3 / 1000.0
        
        # Количество вокселей
        gt_voxels = np.sum(gt)
        pred_voxels = np.sum(pred)
        
        # Объемы
        gt_volume_mm3 = gt_voxels * voxel_volume_mm3
        pred_volume_mm3 = pred_voxels * voxel_volume_mm3
        
        gt_volume_ml = gt_volume_mm3 / 1000.0
        pred_volume_ml = pred_volume_mm3 / 1000.0
        
        # Абсолютная разница объемов
        volume_diff_abs = abs(pred_volume_ml - gt_volume_ml)
        
        # Относительная разница объемов
        if gt_volume_ml > 0:
            volume_diff_rel = (abs(pred_volume_ml - gt_volume_ml) / gt_volume_ml) * 100.0
        else:
            volume_diff_rel = 0.0 if pred_volume_ml == 0 else float('inf')
        
        return {
            'volume_gt_ml': float(gt_volume_ml),
            'volume_pred_ml': float(pred_volume_ml),
            'volume_diff_abs_ml': float(volume_diff_abs),
            'volume_diff_rel_percent': float(volume_diff_rel),
            'voxels_gt': int(gt_voxels),
            'voxels_pred': int(pred_voxels),
            'voxel_volume_mm3': float(voxel_volume_mm3)
        }
    
    def volume_similarity(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        Сходство объемов
        
        VS = 1 - |V_pred - V_gt| / (V_pred + V_gt)
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Коэффициент сходства объемов [0, 1]
        """
        volume_metrics = self.volume_metrics(ground_truth, prediction)
        
        v_gt = volume_metrics['volume_gt_ml']
        v_pred = volume_metrics['volume_pred_ml']
        
        if v_gt + v_pred == 0:
            return 1.0
        
        vs = 1.0 - (abs(v_pred - v_gt) / (v_pred + v_gt))
        return float(vs)
    
    # ===========================================================================
    # 5. CLINICAL METRICS (Клинические метрики)
    # ===========================================================================
    
    def confusion_matrix_metrics(self, ground_truth: ArrayLike, prediction: ArrayLike) -> Dict[str, float]:
        """
        Метрики на основе confusion matrix
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Словарь с клиническими метриками
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth)
        pred = self._ensure_binary(prediction)
        
        # Вычисляем TP, TN, FP, FN
        tp = np.sum((gt == 1) & (pred == 1))
        tn = np.sum((gt == 0) & (pred == 0))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        
        # Чувствительность (Recall, True Positive Rate)
        sensitivity = tp / (tp + fn + self.config.epsilon)
        
        # Специфичность (True Negative Rate)
        specificity = tn / (tn + fp + self.config.epsilon)
        
        # Точность (Precision, Positive Predictive Value)
        precision = tp / (tp + fp + self.config.epsilon)
        
        # Отрицательная предсказательная ценность
        npv = tn / (tn + fn + self.config.epsilon) if (tn + fn) > 0 else 0.0
        
        # F1-score (гармоническое среднее precision и recall)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + self.config.epsilon)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.config.epsilon)
        
        # Коэффициент корреляции Мэттьюса (бинарный случай)
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / (mcc_denominator + self.config.epsilon)
        
        return {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'negative_predictive_value': float(npv),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'matthews_correlation': float(mcc)
        }
    
    # ===========================================================================
    # 6. BOUNDARY METRICS (Метрики границ)
    # ===========================================================================
    
    def boundary_iou(self, ground_truth: ArrayLike, prediction: ArrayLike) -> float:
        """
        IoU для граничных регионов
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Boundary IoU [0, 1]
        """
        self._validate_inputs(ground_truth, prediction)
        
        gt = self._ensure_binary(ground_truth)
        pred = self._ensure_binary(prediction)
        
        # Выделяем граничные регионы
        gt_boundary = self._get_boundary_region(gt)
        pred_boundary = self._get_boundary_region(pred)
        
        # Вычисляем IoU для границ
        intersection = np.sum(gt_boundary & pred_boundary)
        union = np.sum(gt_boundary | pred_boundary)
        
        boundary_iou = intersection / (union + self.config.epsilon)
        return float(boundary_iou)
    
    def _get_boundary_region(self, mask: np.ndarray, dilation: int = None) -> np.ndarray:
        """Получение граничного региона с дилатацией"""
        if dilation is None:
            dilation = self.config.surface_dilation
        
        # Выделяем границы
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure=structure)
        boundary = mask ^ eroded
        
        # Применяем дилатацию для расширения граничного региона
        if dilation > 0:
            boundary = ndimage.binary_dilation(boundary, structure=structure, iterations=dilation)
        
        return boundary.astype(np.uint8)
    
    # ===========================================================================
    # 7. QUALITY METRICS (Метрики качества)
    # ===========================================================================
    
    def clinical_quality_assessment(self, ground_truth: ArrayLike, prediction: ArrayLike) -> Dict[str, any]:
        """
        Клиническая оценка качества сегментации
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
        
        Returns:
            Словарь с клинической оценкой
        """
        dice = self.dice_coefficient(ground_truth, prediction)
        hd95 = self.hausdorff_distance_95(ground_truth, prediction)
        volume_metrics = self.volume_metrics(ground_truth, prediction)
        
        # Определяем качество на основе Dice
        if dice >= 0.95:
            quality_grade = "Excellent"
            clinical_acceptable = True
        elif dice >= 0.90:
            quality_grade = "Very Good"
            clinical_acceptable = True
        elif dice >= 0.85:
            quality_grade = "Good"
            clinical_acceptable = True
        elif dice >= 0.80:
            quality_grade = "Moderate"
            clinical_acceptable = True
        else:
            quality_grade = "Poor"
            clinical_acceptable = False
        
        # Проверка объема печени
        volume_ml = volume_metrics['volume_pred_ml']
        if volume_ml < self.config.min_liver_volume_ml:
            volume_warning = "Volume too small"
            clinical_acceptable = False
        elif volume_ml > self.config.max_liver_volume_ml:
            volume_warning = "Volume too large"
            clinical_acceptable = False
        else:
            volume_warning = "Volume within normal range"
        
        # Проверка расстояния Хаусдорфа
        if hd95 > 10.0:  # 10 мм
            distance_warning = "Large surface errors"
            clinical_acceptable = False
        else:
            distance_warning = "Surface accuracy acceptable"
        
        return {
            'quality_grade': quality_grade,
            'clinical_acceptable': clinical_acceptable,
            'dice_threshold_met': dice >= 0.90,
            'volume_assessment': volume_warning,
            'distance_assessment': distance_warning,
            'recommendation': "Accept" if clinical_acceptable else "Review needed"
        }
    
    # ===========================================================================
    # 8. COMPREHENSIVE METRICS (Все метрики)
    # ===========================================================================
    
    def calculate_all_metrics(self, ground_truth: ArrayLike, 
                            prediction: ArrayLike,
                            verbose: bool = False) -> Dict[str, any]:
        """
        Расчет всех метрик
        
        Args:
            ground_truth: Истинная маска
            prediction: Предсказанная маска
            verbose: Выводить информацию о времени выполнения
        
        Returns:
            Словарь со всеми метриками
        """
        results = {}
        start_total = time.perf_counter()
        
        try:
            # 1. Overlap Metrics
            if verbose:
                print("Calculating overlap metrics...")
            
            results['dice'], time_dice = self._timed_execution(
                self.dice_coefficient, ground_truth, prediction
            )
            results['iou'], time_iou = self._timed_execution(
                self.jaccard_index, ground_truth, prediction
            )
            results['volume_overlap_error'], time_voe = self._timed_execution(
                self.volume_overlap_error, ground_truth, prediction
            )
            
            # 2. Statistical Metrics
            if verbose:
                print("Calculating statistical metrics...")
            
            results['mae'], time_mae = self._timed_execution(
                self.mean_absolute_error, ground_truth, prediction
            )
            results['mse'], time_mse = self._timed_execution(
                self.mean_squared_error, ground_truth, prediction
            )
            results['rmse'], time_rmse = self._timed_execution(
                self.root_mean_squared_error, ground_truth, prediction
            )
            results['normalized_mae'], time_nmae = self._timed_execution(
                self.normalized_mae, ground_truth, prediction
            )
            
            # 3. Distance Metrics
            if verbose:
                print("Calculating distance metrics...")
            
            results['hausdorff_distance'], time_hd = self._timed_execution(
                self.hausdorff_distance, ground_truth, prediction
            )
            results['hausdorff_distance_95'], time_hd95 = self._timed_execution(
                self.hausdorff_distance_95, ground_truth, prediction
            )
            results['average_surface_distance'], time_asd = self._timed_execution(
                self.average_surface_distance, ground_truth, prediction
            )
            
            # 4. Volume Metrics
            if verbose:
                print("Calculating volume metrics...")
            
            volume_results, time_volume = self._timed_execution(
                self.volume_metrics, ground_truth, prediction
            )
            results.update(volume_results)
            
            results['volume_similarity'], time_vs = self._timed_execution(
                self.volume_similarity, ground_truth, prediction
            )
            
            # 5. Clinical Metrics
            if verbose:
                print("Calculating clinical metrics...")
            
            clinical_results, time_clinical = self._timed_execution(
                self.confusion_matrix_metrics, ground_truth, prediction
            )
            results.update(clinical_results)
            
            # 6. Boundary Metrics
            if verbose:
                print("Calculating boundary metrics...")
            
            results['boundary_iou'], time_boundary = self._timed_execution(
                self.boundary_iou, ground_truth, prediction
            )
            
            # 7. Quality Assessment
            if verbose:
                print("Performing quality assessment...")
            
            quality_results, time_quality = self._timed_execution(
                self.clinical_quality_assessment, ground_truth, prediction
            )
            results.update(quality_results)
            
            # 8. Execution Times
            if verbose:
                total_time = time.perf_counter() - start_total
                execution_times = {
                    'dice': time_dice,
                    'iou': time_iou,
                    'mae': time_mae,
                    'mse': time_mse,
                    'hd95': time_hd95,
                    'asd': time_asd,
                    'volume': time_volume,
                    'clinical': time_clinical,
                    'total': total_time
                }
                results['execution_times'] = execution_times
                
                print(f"\nExecution times:")
                for metric, exec_time in execution_times.items():
                    if metric != 'total':
                        print(f"  {metric.upper()}: {exec_time:.4f}s")
                print(f"  TOTAL: {total_time:.4f}s")
            
            return results
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return {'error': str(e)}
    
    def calculate_summary_statistics(self, metrics_list: List[Dict]) -> Dict[str, any]:
        """
        Расчет статистики по набору метрик
        
        Args:
            metrics_list: Список словарей с метриками
        
        Returns:
            Словарь со статистикой
        """
        if not metrics_list:
            return {}
        
        summary = {}
        all_keys = set()
        
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(metrics[key])
            
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_median"] = float(np.median(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))
                summary[f"{key}_q25"] = float(np.percentile(values, 25))
                summary[f"{key}_q75"] = float(np.percentile(values, 75))
        
        return summary
    
    def generate_report(self, metrics: Dict, format: str = 'text') -> str:
        """
        Генерация отчета по метрикам
        
        Args:
            metrics: Словарь с метриками
            format: Формат отчета ('text', 'html', 'json')
        
        Returns:
            Отчет в указанном формате
        """
        if format == 'json':
            import json
            return json.dumps(metrics, indent=2, ensure_ascii=False)
        
        elif format == 'html':
            html = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    .excellent { color: #27ae60; font-weight: bold; }
                    .good { color: #f39c12; }
                    .poor { color: #e74c3c; }
                </style>
            </head>
            <body>
                <h1>Liver Segmentation Metrics Report</h1>
            """
            
            # Add sections
            sections = [
                ('Overlap Metrics', ['dice', 'iou', 'volume_overlap_error']),
                ('Statistical Metrics', ['mae', 'mse', 'rmse', 'normalized_mae']),
                ('Distance Metrics', ['hausdorff_distance', 'hausdorff_distance_95', 'average_surface_distance']),
                ('Volume Metrics', ['volume_gt_ml', 'volume_pred_ml', 'volume_diff_abs_ml', 'volume_similarity']),
                ('Clinical Metrics', ['sensitivity', 'specificity', 'precision', 'f1_score', 'accuracy']),
            ]
            
            for section_name, metric_keys in sections:
                html += f"<h2>{section_name}</h2><table>"
                html += "<tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>"
                
                for key in metric_keys:
                    if key in metrics:
                        value = metrics[key]
                        assessment = self._assess_metric(key, value)
                        html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.4f}</td><td class='{assessment['class']}'>{assessment['text']}</td></tr>"
                
                html += "</table>"
            
            html += "</body></html>"
            return html
        
        else:  # text format
            report = []
            report.append("=" * 80)
            report.append("LIVER SEGMENTATION METRICS REPORT")
            report.append("=" * 80)
            
            sections = [
                ('📊 OVERLAP METRICS', ['dice', 'iou', 'volume_overlap_error']),
                ('📈 STATISTICAL METRICS', ['mae', 'mse', 'rmse', 'normalized_mae']),
                ('📏 DISTANCE METRICS', ['hausdorff_distance', 'hausdorff_distance_95', 'average_surface_distance']),
                ('🧪 VOLUME METRICS', ['volume_gt_ml', 'volume_pred_ml', 'volume_diff_abs_ml', 'volume_similarity']),
                ('🏥 CLINICAL METRICS', ['sensitivity', 'specificity', 'precision', 'f1_score', 'accuracy']),
                ('✅ QUALITY ASSESSMENT', ['quality_grade', 'clinical_acceptable', 'recommendation']),
            ]
            
            for section_name, metric_keys in sections:
                report.append(f"\n{section_name}")
                report.append("-" * 40)
                
                for key in metric_keys:
                    if key in metrics:
                        value = metrics[key]
                        if isinstance(value, bool):
                            display_value = "✓ YES" if value else "✗ NO"
                        elif isinstance(value, str):
                            display_value = value
                        else:
                            display_value = f"{value:.4f}"
                        
                        report.append(f"  {key.replace('_', ' ').title():30s}: {display_value}")
            
            report.append("\n" + "=" * 80)
            return "\n".join(report)
    
    def _assess_metric(self, metric_name: str, value: float) -> Dict[str, str]:
        """Оценка метрики"""
        assessments = {
            'dice': [
                (0.95, 'excellent', 'Excellent'),
                (0.90, 'good', 'Good'),
                (0.85, 'moderate', 'Moderate'),
                (0.0, 'poor', 'Poor')
            ],
            'mae': [
                (0.01, 'excellent', 'Excellent'),
                (0.05, 'good', 'Good'),
                (0.10, 'moderate', 'Moderate'),
                (1.0, 'poor', 'Poor')
            ],
            'hausdorff_distance_95': [
                (2.0, 'excellent', '< 2mm'),
                (5.0, 'good', '< 5mm'),
                (10.0, 'moderate', '< 10mm'),
                (100.0, 'poor', '≥ 10mm')
            ],
            'volume_similarity': [
                (0.98, 'excellent', 'Excellent'),
                (0.95, 'good', 'Good'),
                (0.90, 'moderate', 'Moderate'),
                (0.0, 'poor', 'Poor')
            ],
        }
        
        if metric_name not in assessments:
            return {'class': '', 'text': ''}
        
        for threshold, css_class, text in assessments[metric_name]:
            if value >= threshold:
                return {'class': css_class, 'text': text}
        
        return {'class': 'poor', 'text': 'Poor'}


# ===========================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ===========================================================================

def main():
    """Демонстрация работы метрик"""
    print("=" * 80)
    print("Advanced Medical Image Segmentation Metrics")
    print("=" * 80)
    
    # Создаем тестовые данные
    np.random.seed(42)
    shape = (64, 128, 128)
    
    # Ground truth - сферическая область
    gt = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) // 2
    radius = 30
    
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    gt[distance <= radius] = 1.0
    
    # Prediction - слегка смещенная и искаженная область
    pred = np.zeros(shape, dtype=np.float32)
    center_pred = center + np.array([2, 3, -1])
    radius_pred = radius - 2
    
    distance_pred = np.sqrt((z - center_pred[0])**2 + 
                           (y - center_pred[1])**2 + 
                           (x - center_pred[2])**2)
    pred[distance_pred <= radius_pred] = 1.0
    
    # Добавляем немного шума
    noise = np.random.randn(*shape) * 0.1
    pred = np.clip(pred + noise, 0, 1)
    
    # Конфигурация с реальными параметрами КТ
    config = MetricConfig(
        spacing=(1.5, 0.98, 0.98),  # типичные параметры КТ
        hausdorff_percentile=95.0,
        min_liver_volume_ml=800,
        max_liver_volume_ml=2500
    )
    
    # Создаем экземпляр метрик
    metrics_calculator = SegmentationMetrics(config)
    
    print("\n📊 Расчет всех метрик...")
    print("-" * 40)
    
    # Расчет всех метрик
    all_metrics = metrics_calculator.calculate_all_metrics(gt, pred, verbose=True)
    
    print("\n📋 Краткий отчет:")
    print("-" * 40)
    
    # Важные метрики
    important_metrics = ['dice', 'mae', 'hausdorff_distance_95', 
                        'volume_pred_ml', 'sensitivity', 'specificity']
    
    for metric in important_metrics:
        if metric in all_metrics:
            value = all_metrics[metric]
            if isinstance(value, float):
                print(f"{metric.upper():25s}: {value:.4f}")
            else:
                print(f"{metric.upper():25s}: {value}")
    
    print("\n📄 Полный отчет:")
    print("-" * 40)
    
    # Генерация текстового отчета
    report = metrics_calculator.generate_report(all_metrics, format='text')
    print(report)
    
    # Сохранение отчета в файл
    with open('segmentation_metrics_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Отчет сохранен в 'segmentation_metrics_report.txt'")
    
    # Дополнительные тесты
    print("\n🧪 Дополнительные тесты:")
    print("-" * 40)
    
    # Тест MAE на идеальном случае
    perfect_mae = metrics_calculator.mean_absolute_error(gt, gt)
    print(f"MAE для идеального случая (GT vs GT): {perfect_mae:.6f}")
    
    # Тест MAE на противоположном случае
    opposite = 1 - gt
    worst_mae = metrics_calculator.mean_absolute_error(gt, opposite)
    print(f"MAE для противоположного случая: {worst_mae:.6f}")
    
    # Объемные метрики
    volume_info = metrics_calculator.volume_metrics(gt, pred)
    print(f"\nОбъем печени (GT): {volume_info['volume_gt_ml']:.1f} мл")
    print(f"Объем печени (Pred): {volume_info['volume_pred_ml']:.1f} мл")
    print(f"Абсолютная разница: {volume_info['volume_diff_abs_ml']:.1f} мл")
    print(f"Относительная разница: {volume_info['volume_diff_rel_percent']:.1f}%")
    
    # Качество сегментации
    quality = metrics_calculator.clinical_quality_assessment(gt, pred)
    print(f"\nОценка качества: {quality['quality_grade']}")
    print(f"Клинически приемлемо: {'Да' if quality['clinical_acceptable'] else 'Нет'}")
    print(f"Рекомендация: {quality['recommendation']}")
    
    print("\n" + "=" * 80)
    print("✅ Демонстрация завершена успешно!")
    print("=" * 80)


if __name__ == "__main__":
    # Проверка наличия необходимых библиотек
    try:
        import scipy
        import scipy.spatial
        main()
    except ImportError as e:
        print(f"Ошибка: необходимо установить библиотеки: {e}")
        print("Установите: pip install scipy numpy")
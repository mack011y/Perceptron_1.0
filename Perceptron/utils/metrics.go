package utils

import (
	"Perceptron/model"
)

// MetricsConfig содержит параметры для расчета метрик
type MetricsConfig struct {
	// Порог для бинаризации вероятностей
	Threshold float64
	
	// Нужно ли вычислять дополнительные метрики (precision, recall, f1)
	ExtendedMetrics bool
}

// DefaultMetricsConfig создает конфигурацию метрик с настройками по умолчанию
func DefaultMetricsConfig() *MetricsConfig {
	return &MetricsConfig{
		Threshold:       0.5,
		ExtendedMetrics: false,
	}
}

// EvaluationResult содержит результаты оценки модели
type EvaluationResult struct {
	Accuracy  float64     // Точность классификации
	CM        [2][2]int   // Матрица ошибок [TN, FP; FN, TP]
	Precision float64     // Точность (Precision)
	Recall    float64     // Полнота (Recall)
	F1        float64     // F1-мера
}

// Accuracy вычисляет точность модели
func Accuracy(p *model.Perceptron, X [][]float64, y []float64, config *MetricsConfig) float64 {
	if config == nil {
		config = DefaultMetricsConfig()
	}
	
	predictions := p.PredictBatch(X)
	
	// Переводим вероятности в бинарные метки
	correct := 0
	for i, pred := range predictions {
		binaryPred := 0.0
		if pred > config.Threshold {
			binaryPred = 1.0
		}
		
		if binaryPred == y[i] {
			correct++
		}
	}
	
	return 100.0 * float64(correct) / float64(len(y))
}

// ConfusionMatrix вычисляет матрицу ошибок
func ConfusionMatrix(p *model.Perceptron, X [][]float64, y []float64, config *MetricsConfig) [2][2]int {
	if config == nil {
		config = DefaultMetricsConfig()
	}
	
	predictions := p.PredictBatch(X)
	cm := [2][2]int{} // [true][pred]
	
	for i, pred := range predictions {
		trueClass := int(y[i])
		predClass := 0
		if pred > config.Threshold {
			predClass = 1
		}
		
		cm[trueClass][predClass]++
	}
	
	return cm
}

// EvaluateModel производит полную оценку модели и возвращает набор метрик
func EvaluateModel(p *model.Perceptron, X [][]float64, y []float64, config *MetricsConfig) *EvaluationResult {
	if config == nil {
		config = DefaultMetricsConfig()
	}
	
	// Вычисляем матрицу ошибок
	cm := ConfusionMatrix(p, X, y, config)
	
	// Извлекаем компоненты матрицы ошибок
	tn, fp := cm[0][0], cm[0][1]
	fn, tp := cm[1][0], cm[1][1]
	
	// Рассчитываем основные метрики
	total := float64(tn + fp + fn + tp)
	accuracy := 100.0 * float64(tn+tp) / total
	
	// Инициализируем результат
	result := &EvaluationResult{
		Accuracy: accuracy,
		CM:       cm,
	}
	
	// Вычисляем дополнительные метрики, если требуется
	if config.ExtendedMetrics {
		// Precision = TP / (TP + FP)
		if tp+fp > 0 {
			result.Precision = float64(tp) / float64(tp+fp)
		}
		
		// Recall = TP / (TP + FN)
		if tp+fn > 0 {
			result.Recall = float64(tp) / float64(tp+fn)
		}
		
		// F1 = 2 * Precision * Recall / (Precision + Recall)
		if result.Precision+result.Recall > 0 {
			result.F1 = 2 * result.Precision * result.Recall / (result.Precision + result.Recall)
		}
	}
	
	return result
} 
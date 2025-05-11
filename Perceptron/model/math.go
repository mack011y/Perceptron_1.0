package model

// ExpTaylor вычисляет экспоненту через ряд Тейлора
// e^x = 1 + x + x^2/2! + x^3/3! + ... + x^n/n!
func ExpTaylor(x float64) float64 {
	// Ограничения для предотвращения переполнения
	if x > 20.0 {
		return 485165195.4097903 // приблизительно e^20
	}
	if x < -20.0 {
		return 2.0611536224385579e-9 // приблизительно e^-20
	}

	// Используем тождество e^x = (e^(x/2))^2
	// это улучшает точность для больших x
	if x > 5.0 {
		half := ExpTaylor(x * 0.5)
		return half * half
	}
	
	result := 1.0
	term := 1.0
	n := 1.0
	
	// Вычисляем сумму ряда
	for i := 1; i < 25; i++ {
		term *= x / n
		result += term
		n += 1.0
		
		// Если член ряда стал очень маленьким, прекращаем
		if term < 1e-15 * result {
			break
		}
	}
	
	return result
}

// Sigmoid вычисляет точную сигмоидную функцию
// σ(z) = 1 / (1 + e^(-z))
// Используем свойство: σ(-z) = 1 - σ(z)
func Sigmoid(z float64) float64 {
	// Обрабатываем краевые случаи
	if z < -20.0 {
		return 2.0611536224385579e-9
	}
	if z > 20.0 {
		return 0.9999999979388464
	}
	
	// Для отрицательных значений используем свойство σ(-z) = 1 - σ(z)
	if z < 0 {
		return 1.0 - Sigmoid(-z)
	}
	
	// Для положительных значений вычисляем напрямую
	// σ(z) = e^z / (e^z + 1)
	ez := ExpTaylor(z)
	return ez / (ez + 1.0)
}

// Log реализация натурального логарифма
func Log(x float64) float64 {
	// Базовые случаи и защита от некорректных входных значений
	if x <= 0.0 {
		return -1000.0 // Очень большое отрицательное число
	}
	if x == 1.0 {
		return 0.0
	}
	
	// Для значений близких к 1 используем ряд Тейлора
	// ln(1+y) = y - y^2/2 + y^3/3 - ...
	if x >= 0.8 && x <= 1.2 {
		y := x - 1.0
		result := 0.0
		term := y
		n := 1.0
		
		for i := 1; i <= 30; i++ {
			result += term / n
			term *= -y
			n += 1.0
			
			if term / n < 1e-15 && term / n > -1e-15 {
				break
			}
		}
		
		return result
	}
	
	// Для остальных значений используем метод бинарного разложения
	// ln(x) = ln(2^k * y) = k*ln(2) + ln(y), где 1 <= y < 2
	k := 0
	y := x
	
	// Нормализуем y в диапазон [1,2)
	if y < 1.0 {
		for y < 0.5 {
			y *= 2.0
			k--
		}
	} else {
		for y >= 2.0 {
			y /= 2.0
			k++
		}
	}
	
	// Константа ln(2)
	ln2 := 0.6931471805599453
	
	// Аппроксимация ln(y) для y в диапазоне [0.5, 2)
	// Используем ряд Тейлора вокруг 1
	t := y - 1.0
	result := t
	tPower := t * t
	sign := -1.0
	
	for n := 2.0; n <= 30.0; n++ {
		term := sign * tPower / n
		result += term
		tPower *= t
		sign *= -1.0
		
		if term < 0 {
			term = -term
		}
		
		if term < 1e-15 {
			break
		}
	}
	
	return float64(k)*ln2 + result
}

// CrossEntropy вычисляет функцию потери бинарной кросс-энтропии
func CrossEntropy(y, yPred []float64) float64 {
	sum := 0.0
	for i := 0; i < len(y); i++ {
		// Защита от 0 и 1 для логарифма
		prediction := 0.0
		if yPred[i] < 1e-15 {
			prediction = 1e-15
		} else if yPred[i] > 1.0-1e-15 {
			prediction = 1.0 - 1e-15
		} else {
			prediction = yPred[i]
		}
				
		sum += y[i]*Log(prediction) + (1.0-y[i])*Log(1.0 - prediction)
	}
	
	return -sum / float64(len(y))
} 
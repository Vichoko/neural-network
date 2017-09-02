package util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import network.NeuralNetwork;
import util.utils.E_SamplingMethod;
/**
 * Utilidades varias para el funcionamiento de la red neuronal.
 * @author vichoko
 *
 */
public class utils {
	
	public enum E_SamplingMethod {
		SUBSAMPLE, OVERSAMPLE
	}
	/**
	 * Genera metricas para una red neuronal, a partir de datos de prueba; las cuales son retornadas en un HashMap. 
	 * Las metricas disponibles se pueden obtener con las llaves:
	 * "tasa_aciertos", "tasa_desaciertos", "precision" y "recall".
	 * 
	 * Para predicciones binarias en las cuales hay solo 1 output y un threshold desde el cual se interperta como clase 1.
	 * 
	 * @param net Red que se desea evaluar.
	 * @param input Datos de entrada de prueba.
	 * @param expectedOutput Datos de salida esperados para los datos de prueba.
	 * @param threshold Umbral para separar clases binarias (entre 0 y 1).
	 * @param verbose Valor booleano para activar las impresiones en consola.
	 * @return HashMap que contiene metricas, grabadas por su nombre clave.
	 * @throws Exception Si entrada y salida esperada tienen distintas salidas.
	 */
	public static HashMap<String, Double> binaryMetrics(NeuralNetwork net, double[][] input, double[][] expectedOutput, double threshold, boolean verbose) throws Exception {
		if (input.length != expectedOutput.length) {
			throw new Exception("utils.metrics :: input & expectedOutput have different sizes.");
		}
		// Se almacenan metricas en hasmap, por el nombre de la metrica.
		HashMap<String, Double> metricsData = new HashMap<>();
		
		// revisar balance de clases
		int class1counter = 0;
		int class0counter = 0;
		for (double[] clas : expectedOutput) {
			if (clas[0] == 1) {
				class1counter++;
			} else {
				class0counter++;
			}
		}
		if (verbose) {
			System.out.println("Test data class balance:");
			System.out.println("class 0: " + class0counter);
			System.out.println("class 1: " + class1counter);
		}
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		for (int i = 0; i < input.length; i++) {
			int[] realOutput = net.binaryPredict(input[i], threshold);
			// expected class is 0
			if (expectedOutput[i][0] == 0 && 0 == realOutput[0]) 
				verdaderosNegativos++;
			else if (expectedOutput[i][0] == 0 && 1 == realOutput[0]) 
				falsosPositivos++;
			else if (expectedOutput[i][0] == 1 && 1 == realOutput[0])
				verdaderosPositivos++;
			else if (expectedOutput[i][0] == 1 && 0 == realOutput[0])
				falsosNegativos++;
			else
				System.out.println("expected is "+expectedOutput[i][0]+" real is "+realOutput[0]);
		}
		
	
		double tasaAciertos = (verdaderosPositivos + verdaderosNegativos)*1.0/input.length;
		double tasaDesaciertos = (falsosPositivos + falsosNegativos)*1.0/input.length;
		
		metricsData.put("tasa_aciertos", tasaAciertos);
		metricsData.put("tasa_desaciertos", tasaDesaciertos);
		metricsData.put("precision", verdaderosPositivos*1.0/(verdaderosPositivos+falsosPositivos));
		metricsData.put("recall", verdaderosPositivos*1.0/(verdaderosPositivos+falsosNegativos));
		
		if (verbose) {
			System.out.println();
			System.out.println("Metricas de desempeño de la red neuronal: ");
			System.out.println("Numero de experimentos: " + input.length);
			System.out.println("Verdaderos Positivos: " + verdaderosPositivos);
			System.out.println("Verdaderos Negativos: " + verdaderosNegativos);
			System.out.println("Falsos Positivos: " + falsosPositivos);
			System.out.println("Falsos Negativos: " + falsosNegativos);
			System.out.println("Tasa aciertos: " + tasaAciertos + "; tasa desaciertos: " + tasaDesaciertos);
			System.out.println("Precision: " + metricsData.get("precision"));
			System.out.println("Recall: " + metricsData.get("recall"));
				
		}
		
		return metricsData;
	}
	
	/**
	 * Genera metricas para una red neuronal, a partir de datos de prueba; las cuales son retornadas en un HashMap. 
	 * Las metricas disponibles se pueden obtener con las llaves:
	 * "tasa_aciertos", "tasa_desaciertos", "precision" y "recall".
	 * 
	 * Para predicciones binarias en las cuales hay 2 output, y el indice del output predominante determina la clase (1 o 0)
	 * 
	 * @param net Red que se desea evaluar.
	 * @param input Datos de entrada de prueba.
	 * @param expectedOutput Datos de salida esperados para los datos de prueba.
	 * @param verbose Valor booleano para activar las impresiones en consola.
	 * @return HashMap que contiene metricas, grabadas por su nombre clave.
	 * @throws Exception Si entrada y salida esperada tienen distintas salidas.
	 */
	public static HashMap<String, Double> binaryDualOutputMetrics(NeuralNetwork net, double[][] input, double[][] expectedOutput, boolean verbose) throws Exception {
		if (input.length != expectedOutput.length) {
			throw new Exception("utils.metrics :: input & expectedOutput have different sizes.");
		}
		// Se almacenan metricas en hasmap, por el nombre de la metrica.
		HashMap<String, Double> metricsData = new HashMap<>();
		
		// revisar balance de clases
		int class1counter = 0;
		int class0counter = 0;
		for (double[] clas : expectedOutput) {
			if (clas[0] == 1) {
				class1counter++;
			} else {
				class0counter++;
			}
		}
		if (verbose) {
			System.out.println("Test data class balance:");
			System.out.println("class 0: " + class0counter);
			System.out.println("class 1: " + class1counter);
		}
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		for (int i = 0; i < input.length; i++) {
			double[] realOutput = net.predict(input[i]);
			// expected class is 0
			if (expectedOutput[i][0] >= expectedOutput[i][1] && realOutput[0] >= realOutput[1]) 
				verdaderosNegativos++;
			else if (expectedOutput[i][0] >= expectedOutput[i][1] && realOutput[0] < realOutput[1]) 
				falsosPositivos++;
			else if (expectedOutput[i][0] < expectedOutput[i][1] && realOutput[0] < realOutput[1])
				verdaderosPositivos++;
			else if (expectedOutput[i][0] < expectedOutput[i][1] && realOutput[0] >= realOutput[1])
				falsosNegativos++;
			else
				System.out.println("expected is "+expectedOutput[i][0]+" real is "+realOutput[0]);
		}
		
	
		double tasaAciertos = (verdaderosPositivos + verdaderosNegativos)*1.0/input.length;
		double tasaDesaciertos = (falsosPositivos + falsosNegativos)*1.0/input.length;
		
		metricsData.put("tasa_aciertos", tasaAciertos);
		metricsData.put("tasa_desaciertos", tasaDesaciertos);
		metricsData.put("precision", verdaderosPositivos*1.0/(verdaderosPositivos+falsosPositivos));
		metricsData.put("recall", verdaderosPositivos*1.0/(verdaderosPositivos+falsosNegativos));
		
		if (verbose) {
			System.out.println();
			System.out.println("Metricas de desempeño de la red neuronal: ");
			System.out.println("Numero de experimentos: " + input.length);
			System.out.println("Verdaderos Positivos: " + verdaderosPositivos);
			System.out.println("Verdaderos Negativos: " + verdaderosNegativos);
			System.out.println("Falsos Positivos: " + falsosPositivos);
			System.out.println("Falsos Negativos: " + falsosNegativos);
			System.out.println("Tasa aciertos: " + tasaAciertos + "; tasa desaciertos: " + tasaDesaciertos);
			System.out.println("Precision: " + metricsData.get("precision"));
			System.out.println("Recall: " + metricsData.get("recall"));
				
		}
		
		return metricsData;
	}
/**
 * Normaliza el input, para llevarlos a valores entre 0 y 1.
 * @param inputs vector de entrada que se desea normalizar
 * @param minVal valor minimo dentro del vector de entrada
 * @param maxVal valor maximo dentro del vector de entrada
 * @return arreglo de entradas con valores normalizados entre 0 y 1
 */
	public static double[][] normalize(double[][] inputs, double minVal, double maxVal) {
		double[][] res = new double[inputs.length][];
		int inputIndex = 0;
		for (double[] vector : inputs) {
			
			int vectorIndex = 0;
			double[] normalizedVector = new double[vector.length];
			for (double value : vector) {
				normalizedVector[vectorIndex++] = (value - minVal)/(maxVal - minVal);
			}	
			res[inputIndex++] = normalizedVector;
		}
		return res;
	}
	
	public static double[][] normalizeAny(double[][] in){
		double minVal = Double.MAX_VALUE;
		double maxVal = Double.MIN_VALUE;
		for (double[] X : in) {
	        for (double val : X) {
		        if (val > maxVal) {
		        	maxVal = val;
		        }
		        if (minVal > val) {
		        	minVal = val;
		        }	
	        }
		}		
		return normalize(in, minVal, maxVal);
	}
	/**
	 * Aplica operacion inversa a normalizacion.
	 * @param inputs 
	 * @param minVal
	 * @param maxVal
	 * @return
	 */
	public static double[][] denormalize(double[][] inputs, double minVal, double maxVal) {
		double[][] res = new double[inputs.length][];
		int inputIndex = 0;
		for (double[] vector : inputs) {
			double[] normalizedVector = new double[vector.length];
			int vectorIndex = 0;
			for (double value : vector){
				normalizedVector[vectorIndex++] = ((minVal - maxVal)*value - minVal)/(-1);
			}
			res[inputIndex++] = normalizedVector;
		}
		return res;
	}
	

	
	/** METODOS DEL PAQUETE NETWORK */
	public static double transferDerivative(double output) {
		return output * (1.0-output);
	}
	
	
	public static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}

	public static double dotProduct(double[] weigths, double[] inputs) throws Exception {
		if (weigths.length != inputs.length) {
			throw new Exception("weigth and inputs vectors have different lengths.");
		}
		double res = 0;
		for (int i = 0; i < weigths.length; i++) {
			res += weigths[i]*inputs[i];
		}
		return res;
	}
	/**
	 * Metodo para post-procesar los datos. Util en caso de desbalance de clases, se puede hacer oversampling o subsampling. 
	 * Ademas al final distribuye los datos intercalados por clases.
	 * @param data Datos de entrada para el clasificador.
	 * @param classes Clases esperadas para el clasificador.
	 * @param sampling Tipo de sampling que se desea hacer.
	 */
	public static Object[] sampleData(double[][] data, double[][] classes, utils.E_SamplingMethod sampling) {
		int sampleSize;
		double[][] classes_0;
		double[][] data_0;
	
		double[][] classes_1;
		double[][] data_1;
	
		// Balancear clases
		// listas de indices
		List<Integer> class_1 = new ArrayList<>();
		List<Integer> class_0 = new ArrayList<>();
		int index = 0;
		for (double[] y : classes) {
			if (y[0] == 1.0) {
				class_1.add(index++);
			} else {
				class_0.add(index++);
			}
		}
	
		Random rand = new Random(); // randomizer
		if (sampling == utils.E_SamplingMethod.OVERSAMPLE) {
			// Repetir datos para clases sub-balanceadas
			sampleSize = Math.max(class_1.size(), class_0.size());
			classes_0 = new double[sampleSize][];
			classes_1 = new double[sampleSize][];
			data_0 = new double[sampleSize][];
			data_1 = new double[sampleSize][];
	
			int sampleIndex = 0;
			for (int dataIndex : class_1) {
				classes_1[sampleIndex] = classes[dataIndex];
				data_1[sampleIndex++] = data[dataIndex];
			}
			while (sampleIndex <= sampleSize - 1) {
				// oversample si es necesario
				int randomIndex = rand.nextInt(class_1.size());
				classes_1[sampleIndex] = classes[class_1.get(randomIndex)];
				data_1[sampleIndex++] = data[class_1.get(randomIndex)];
			}
			sampleIndex = 0;
			for (int dataIndex : class_0) {
				classes_0[sampleIndex] = classes[dataIndex];
				data_0[sampleIndex++] = data[dataIndex];
			}
			while (sampleIndex <= sampleSize - 1) {
				// oversample si es necesario
				int randomIndex = rand.nextInt(class_0.size());
				classes_0[sampleIndex] = classes[class_0.get(randomIndex)];
				data_0[sampleIndex++] = data[class_0.get(randomIndex)];
			}
		} else {
			// subsample
			sampleSize = Math.min(class_1.size(), class_0.size());
			classes_0 = new double[sampleSize][];
			classes_1 = new double[sampleSize][];
			data_0 = new double[sampleSize][];
			data_1 = new double[sampleSize][];
	
			int sampleIndex = 0;
			int classIndex = 0;
			while (sampleIndex <= sampleSize - 1) {
				// subsample
				classes_1[sampleIndex] = classes[class_1.get(classIndex)];
				data_1[sampleIndex++] = data[class_1.get(classIndex++)];
			}
			sampleIndex = 0;
			classIndex = 0;
			while (sampleIndex <= sampleSize - 1) {
				// subsample
				classes_0[sampleIndex] = classes[class_0.get(classIndex)];
				data_0[sampleIndex++] = data[class_0.get(classIndex++)];
			}
		}
		// intercalar datos en arreglo final
		data = new double[sampleSize * 2][];
		classes = new double[sampleSize * 2][];
		int class0Index = 0;
		int class1Index = 0;
		for (int i = 0; i < sampleSize * 2; i++) {
			if ((i & 1) == 0) {
				// par
				data[i] = data_0[class0Index];
				classes[i] = classes_0[class0Index++];
			} else {
				data[i] = data_1[class1Index];
				classes[i] = classes_1[class1Index++];
			}
		}
	
		return new Object[] {data, classes};
	}
}

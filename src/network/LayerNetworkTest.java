package network;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.HashMap;

import org.junit.Before;
import org.junit.Test;

import util.utils;

public class LayerNetworkTest {

	NeuralNetwork little;
	NeuralNetwork bigger;
	@Before
	public void setUp() throws Exception {
		little = new NeuralNetwork();
		little.newInputLayer(2, 1);
		little.closeNetwork();
		
		bigger = new NeuralNetwork();
		bigger.newInputLayer(3,2);
		bigger.newHiddenLayer(5);
		bigger.newHiddenLayer(10);
		bigger.closeNetwork();
		
	}
	@Test
	public void LayerTest() throws Exception {
		NeuralLayer first = new NeuralLayer(1,1);
		NeuralLayer second = new NeuralLayer(2,2);
		assertEquals(first.getOutputSize(), 1);
		assertEquals(second.getOutputSize(), 2);
		double[] fres = first.synapsis(new double[] {1});
		double[] sres = second.synapsis(new double[] {2, 3});
		assertEquals(first.getPastOutputs().length, first.getOutputSize());
		assertEquals(second.getPastOutputs().length, second.getOutputSize());
		assertEquals(first.getPastOutputs(), fres);
		assertEquals(second.getPastOutputs(), sres);
		}

	@Test
	public void NetworkTest() throws Exception {
		assertEquals(little.forwardFeed(new double[] {1,2}).length, 1);
		assertEquals(bigger.forwardFeed(new double[] {1,2,4}).length, 10);
		bigger.backPropagation(new double[] {1,2,3,4,5,6,7,8,9,10});
		little.backPropagation(new double[] {1});
	}
	
	
	public void networkLogicGateSingleOutputTest(double[][] input, double[][] expectedOutput, String gateName, boolean verbose) throws Exception {
		NeuralNetwork net = new NeuralNetwork(0.1);
		net.newInputLayer(2, 3);
		net.newHiddenLayer(4);
		net.newHiddenLayer(1);
		net.closeNetwork();
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender XOR.
		 * Clases binarias: 
		 * 	1, si bit_1 <GATE> bit_2 == 1; 
		 * 	0 si no.*/
		
		if (verbose) {
			System.out.println();
			System.out.println(gateName + " neural network training test.");
		}
		net.train(input, expectedOutput, 1000000, gateName);
		
		// preparar datos de prueba
		int casosTotales = 300000;
		double[][] testInput = new double[casosTotales][];
		double[][] testOutput = new double[casosTotales][];
		for (int i = 0; i < casosTotales; i++) {
			double seed = Math.random();
			int randomIndex;
			if (seed < 0.25)
				randomIndex = 0;
			else if (seed < 0.5)
				randomIndex = 1;
			else if (seed < 0.75)
				randomIndex = 2;
			else
				randomIndex = 3;
			testInput[i] = input[randomIndex];
			testOutput[i] = expectedOutput[randomIndex];
		}
		
		double threshold = 0.5;
		HashMap<String, Double> metricsData = 
				utils.binaryMetrics(net, testInput, testOutput, threshold, verbose);
			
		assertTrue(metricsData.get("tasa_aciertos") > 0.70);
		assertTrue(metricsData.get("tasa_desaciertos") < 0.20);
	}	
	public void networkLogicGateDoubleOutputTest(double[][] input, double[][] expectedOutput, String gateName, boolean verbose) throws Exception {
		NeuralNetwork net = new NeuralNetwork(0.1);
		net.newInputLayer(2, 3);
		net.newHiddenLayer(4);
		net.newHiddenLayer(2);
		net.closeNetwork();
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender XOR.
		 * Clases binarias: 
		 * 	1, si bit_1 <GATE> bit_2 == 1; 
		 * 	0 si no.*/
		
		if (verbose) {
			System.out.println();
			System.out.println(gateName + " neural network training test.");
		}
		net.train(input, expectedOutput, 1000000, gateName);
		
		// preparar datos de prueba
		int casosTotales = 300000;
		double[][] testInput = new double[casosTotales][];
		double[][] testOutput = new double[casosTotales][];
		for (int i = 0; i < casosTotales; i++) {
			double seed = Math.random();
			int randomIndex;
			if (seed < 0.25)
				randomIndex = 0;
			else if (seed < 0.5)
				randomIndex = 1;
			else if (seed < 0.75)
				randomIndex = 2;
			else
				randomIndex = 3;
			testInput[i] = input[randomIndex];
			testOutput[i] = expectedOutput[randomIndex];
		}
		
		HashMap<String, Double> metricsData = 
				utils.binaryDualOutputMetrics(net, testInput, testOutput, verbose);
			
		assertTrue(metricsData.get("tasa_aciertos") > 0.70);
		assertTrue(metricsData.get("tasa_desaciertos") < 0.20);
	}
	
	
	
	@Test
	public void networkXORSingleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {0};
		expectedOutput[1] = new double[] {1};
		expectedOutput[2] = new double[] {1};
		expectedOutput[3] = new double[] {0};
		
		boolean verbose = true;
		networkLogicGateSingleOutputTest(input, expectedOutput, "XOR", verbose);
		
	}
	@Test
	public void networkANDSingleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {0};
		expectedOutput[1] = new double[] {0};
		expectedOutput[2] = new double[] {0};
		expectedOutput[3] = new double[] {1};
		
		boolean verbose = true;
		networkLogicGateSingleOutputTest(input, expectedOutput, "AND", verbose);
		
	}
	@Test
	public void networkORSingleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {0};
		expectedOutput[1] = new double[] {1};
		expectedOutput[2] = new double[] {1};
		expectedOutput[3] = new double[] {1};
		
		boolean verbose = true;
		networkLogicGateSingleOutputTest(input, expectedOutput, "OR", verbose);
		
	}
	@Test
	public void networkXORDoubleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {1, 0};
		expectedOutput[1] = new double[] {0, 1};
		expectedOutput[2] = new double[] {0, 1};
		expectedOutput[3] = new double[] {1, 0};
		
		boolean verbose = true;
		networkLogicGateDoubleOutputTest(input, expectedOutput, "XOR_d", verbose);
		
	}
	@Test
	public void networkANDDoubleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {1, 0};
		expectedOutput[1] = new double[] {1, 0};
		expectedOutput[2] = new double[] {1, 0};
		expectedOutput[3] = new double[] {0, 1};
		
		boolean verbose = true;
		networkLogicGateDoubleOutputTest(input, expectedOutput, "AND_d", verbose);
		
	}
	@Test
	public void networkORDoubleOutputLearningTest() throws Exception{		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {1, 0};
		expectedOutput[1] = new double[] {0, 1};
		expectedOutput[2] = new double[] {0, 1};
		expectedOutput[3] = new double[] {0, 1};
		
		boolean verbose = true;
		networkLogicGateDoubleOutputTest(input, expectedOutput, "OR_d", verbose);
		
	}
	@Test
	public void networkLinearFunctionSingleOutputLearningTest() throws Exception {
		boolean verbose = true;
		NeuralNetwork net = new NeuralNetwork(0.2);
		net.newInputLayer(2, 2);
		net.newHiddenLayer(3);
		net.newHiddenLayer(1);
		net.closeNetwork();
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender una función linear.
		 * Clases binarias: 1 si el punto (x,y) esta sobre la recta, 0 si no.*/
		
		// generar datos de entrenamiento y test
		int dataSetSize = 10000;
		double[][] input = new double[dataSetSize][2];
		double[][] expectedOutput = new double[dataSetSize][1];
		for (int i = 0; i < dataSetSize; i++) {
			/** Fun : Y = -1*X
			 * Si punto esta abajo+izquierda, clase 0.
			 * Si punto esta arriba+derecha, clase 1.*/
			input[i] = new double[] {Math.random()*100-50, Math.random()*100-50};
			expectedOutput[i] = new double[] {NeuralTest.abovenrightFunction(input[i][0], input[i][1]) ? 1 : 0};						
		}
		// normalizar valores de entrada
		input = utils.normalize(input, -50, 50);
		
		if (verbose) {
			System.out.println();
			System.out.println("Linear Function neural network training test.");
		}
		// entrenar con mitad
		net.train(Arrays.copyOfRange(input, 1, (int)Math.floor(dataSetSize/2)), 
				Arrays.copyOfRange(expectedOutput, 1, (int)Math.floor(dataSetSize/2)), 
				5000,
				"DIAG");
		
		// evaluar con otra mitad disjunta
		double threshold = 0.5;
		HashMap<String, Double> metricsData = utils.binaryMetrics(net, 
				Arrays.copyOfRange(input, 
						(int)Math.floor(dataSetSize/2) + 1, 
						dataSetSize - 1), 
				Arrays.copyOfRange(expectedOutput, 
						(int)Math.floor(dataSetSize/2) + 1, 
						dataSetSize - 1), 
				threshold,
				verbose);

		assertTrue(metricsData.get("tasa_aciertos") > 0.70);
		assertTrue(metricsData.get("tasa_desaciertos") < 0.20);		
	}
	@Test
	public void networkLinearFunctionDoubleOutputLearningTest() throws Exception {
		boolean verbose = true;
		NeuralNetwork net = new NeuralNetwork(0.2);
		net.newInputLayer(2, 2);
		net.newHiddenLayer(3);
		net.newHiddenLayer(2);
		net.closeNetwork();
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender una función linear.
		 * Clases binarias: 1 si el punto (x,y) esta sobre la recta, 0 si no.*/
		
		// generar datos de entrenamiento y test
		int dataSetSize = 10000;
		double[][] input = new double[dataSetSize][2];
		double[][] expectedOutput = new double[dataSetSize][1];
		for (int i = 0; i < dataSetSize; i++) {
			/** Fun : Y = -1*X
			 * Si punto esta abajo+izquierda, clase 0.
			 * Si punto esta arriba+derecha, clase 1.*/
			input[i] = new double[] {Math.random()*100-50, Math.random()*100-50};
			expectedOutput[i] = NeuralTest.abovenrightFunction(input[i][0], input[i][1]) ? new double[] {0,1} : new double[] {1,0};						
		}
		// normalizar valores de entrada
		input = utils.normalize(input, -50, 50);
		
		if (verbose) {
			System.out.println();
			System.out.println("Linear Function double output neural network training test.");
		}
		// entrenar con mitad
		net.train(Arrays.copyOfRange(input, 1, (int)Math.floor(dataSetSize/2)), 
				Arrays.copyOfRange(expectedOutput, 1, (int)Math.floor(dataSetSize/2)), 
				5000,
				"DIAG_d");
		
		// evaluar con otra mitad disjunta
		HashMap<String, Double> metricsData = utils.binaryDualOutputMetrics(net, 
				Arrays.copyOfRange(input, 
						(int)Math.floor(dataSetSize/2) + 1, 
						dataSetSize - 1), 
				Arrays.copyOfRange(expectedOutput, 
						(int)Math.floor(dataSetSize/2) + 1, 
						dataSetSize - 1), 
				verbose);

		assertTrue(metricsData.get("tasa_aciertos") > 0.70);
		assertTrue(metricsData.get("tasa_desaciertos") < 0.20);		
	}
	

}

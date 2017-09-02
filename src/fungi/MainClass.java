package fungi;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import network.NeuralNetwork;
import util.utils;

public class MainClass {
	static int nFeatures = 122;

	public static void main(String[] args) throws Exception {
		String dataFileName = "agaricus-lepiota.data";
		
		double[][][] dataNClasses = extractNumericData(dataFileName);
		
		double[][] data = dataNClasses[0];
		double[][] classes = dataNClasses[1];
		

		/*Object[] dataNClasses2 = utils.sampleData(data, classes, utils.E_SamplingMethod.SUBSAMPLE);
		data = (double[][]) dataNClasses2[0];
		classes = (double[][]) dataNClasses2[1];*/
		
		NeuralNetwork net = new NeuralNetwork(0.2);
		net.newInputLayer(nFeatures, nFeatures);
		net.newHiddenLayer(10);
		net.newHiddenLayer(2);
		
		// 50% datos de entrenamiento, 50% datos de prueba
		if (data.length != classes.length) {
			throw new Exception("parseTSV :: data size != classes size.");
		}
		net.train(data,
				classes, 
				1000, 
				"FUNGI");

		boolean verbose = true;
		HashMap<String, Double> metricsData = utils.binaryDualOutputMetrics(net,
				data,
				classes, 
				verbose);		
	}

	public static double[][][] extractNumericData(String dataFileName) throws Exception{
		double[][] data;
		double[][] classes;
		List<double[]> classesContainer = new ArrayList<>();
		List<double[]> featuresContainer = new ArrayList<>();
		
		BufferedReader in = new BufferedReader(new FileReader(dataFileName));
		String line = in.readLine();
		while (line != null) {
			// for each raw fungi data
			String[] parsed = line.split(",", -1); // archivo 'comma separated'
			// parse clase o etiqueta
			if (parsed[0].equals("p")) {
				// clase 1, tiene output con indice 1 predominante.
				classesContainer.add(new double[] {0, 1});
			} else {
				// clase 0, tiene output con indice 0 predominante.
				classesContainer.add(new double[] {1, 0});	
			}
			
			String[] categoricalFeatures = Arrays.copyOfRange(parsed, 1, parsed.length); // primer elemento es etiqueta, resto caracteristicas
			double[] attributes = parseCategoricalToNumerical(categoricalFeatures);
			featuresContainer.add(attributes);
			line = in.readLine();
		}
		
		if (featuresContainer.size() != classesContainer.size()) {
			throw new Exception("extractNumericData :: features y classes tienen distinta tamaño");
		}
		
		data = new double[classesContainer.size()][];
		classes = new double[classesContainer.size()][];
		for (int i = 0; i < classesContainer.size(); i++) {
			// transformar a arreglos nativos
			data[i] = featuresContainer.get(i);
			classes[i] = classesContainer.get(i);
		}
		
		return new double[][][] {data, classes};
		
	}
	
	public static double[] parseCategoricalToNumerical(String[] parsed) {
		double[] attributes = new double[nFeatures];
		/** cada columna tiene una cantidad variable de valores categoricos. 
		Los que tienen 2 valores se modelan con 1 bit. 
		Los que tienen mas se modelan con 1 bit para cada categoria posible. 
		*/
		
//		1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s = 6
		int columnIndex = 0;
		int startingIndex = 0;
		if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("x")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("k")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+4] = 1;
		}
			
//		2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s = 4
		columnIndex++;
		startingIndex = 5;
		if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+3] = 1;
		}
		
//		3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y = 10
		columnIndex++;
		startingIndex = 9;
		if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("r")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+8] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+9] = 1;
		}
		
		//	4. bruises?: bruises=t,no=f =1
		columnIndex++;
		startingIndex = 19;
		if (parsed[columnIndex].equals("t")) {
			attributes[startingIndex] = 1;
		}
//		5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s = 9
		columnIndex++;
		startingIndex = 20;
		if (parsed[columnIndex].equals("a")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("l")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("m")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+8] = 1;
		}

	//  6. gill-attachment: attached=a,descending=d,free=f,notched=n = 4
		columnIndex++;
		startingIndex = 28;
		if (parsed[columnIndex].equals("a")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("d")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+3] = 1;
		}
//		7. gill-spacing: close=c,crowded=w,distant=d = 3
		columnIndex++;
		startingIndex = 33;
		if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("d")) {
			attributes[startingIndex+2] = 1;
		} 
//		8. gill-size: broad=b,narrow=n  =1
		columnIndex++;
		startingIndex = 36;
		if (parsed[columnIndex].equals("t")) {
			attributes[startingIndex] = 1;
		}
//		9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y = 12
		columnIndex++;
		startingIndex = 37;
		if (parsed[columnIndex].equals("k")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("h")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("r")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex+8] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+9] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+10] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+11] = 1;
		}
//		10. stalk-shape: enlarging=e,tapering=t =1
		columnIndex++;
		startingIndex = 49;
		if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex] = 1;
		}
//		11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? = 7 
		columnIndex++;
		startingIndex = 50;
		if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("z")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("r")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("?")) {
			attributes[startingIndex+6] = 1;
		}
		
//		12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s =4
		columnIndex++;
		startingIndex = 57;
		if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("k")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+3] = 1;
		}
		
//		13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s =4
		columnIndex++;
		startingIndex = 61;
		if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("k")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+3] = 1;
		}
//		14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y = 9
		columnIndex++;
		startingIndex = 65;
		if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+8] = 1;
		}
//		15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y =9
		columnIndex++;
		startingIndex = 74;
		if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+8] = 1;
		}
//		16. veil-type: partial=p,universal=u  =1
		columnIndex++;
		startingIndex = 83;
		if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex] = 1;
		}
//		17. veil-color: brown=n,orange=o,white=w,yellow=y = 4 
		columnIndex++;
		startingIndex = 84;
		if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+3] = 1;
		}
//		18. ring-number: none=n,one=o,two=t = 3
		columnIndex++;
		startingIndex = 88;
		if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("t")) {
			attributes[startingIndex+2] = 1;
		}
//				19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z = 8
		columnIndex++;
		startingIndex = 91;
		if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("e")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("f")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("l")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("z")) {
			attributes[startingIndex+7] = 1;
		}
//		20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y =9 
		columnIndex++;
		startingIndex = 99;
		if (parsed[columnIndex].equals("k")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("b")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("h")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("r")) {
			attributes[startingIndex+4] = 1;
		} else if (parsed[columnIndex].equals("o")) {
			attributes[startingIndex+5] = 1;
		} else if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex+6] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+7] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+8] = 1;
		}
//		21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y =6
		columnIndex++;
		startingIndex = 108;
		if (parsed[columnIndex].equals("a")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("c")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("n")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("s")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("v")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("y")) {
			attributes[startingIndex+3] = 1;
		}
//		22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d =7
		columnIndex++;
		startingIndex = 114;
		if (parsed[columnIndex].equals("g")) {
			attributes[startingIndex+0] = 1;
		} else if (parsed[columnIndex].equals("l")) {
			attributes[startingIndex+1] = 1;
		} else if (parsed[columnIndex].equals("m")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("p")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("u")) {
			attributes[startingIndex+2] = 1;
		} else if (parsed[columnIndex].equals("w")) {
			attributes[startingIndex+3] = 1;
		} else if (parsed[columnIndex].equals("d")) {
			attributes[startingIndex+3] = 1;
		}		
		return attributes;
		
	}
}

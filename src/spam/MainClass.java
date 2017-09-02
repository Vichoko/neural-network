package spam;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.StringTokenizer;
import java.util.TreeSet;

import network.NeuralNetwork;
import util.utils;

public class MainClass {
	static String TFIDF_TEMPFILE_NAME = "temp_tfidf";
	static boolean DEBUG = false;

	public static void main(String[] args) throws Exception {
		String fileName = "SMSSpamCollection";
		parseTSV(fileName, utils.E_SamplingMethod.SUBSAMPLE, DEBUG);
	}

	/** 
	 * Metodo especifico para cargar archivo TSV de la forma: <String clase>	<String SMS>.
	 * Transforma el archivo de texto a datos que puede consumir la red neuronal (double[][] data y double[][] classes).
	 * @param fileName Nombre del archivo de datos tsv
	 * @param sampling Tipo de sampling que se quiere hacer sobre los datos desbalanceados.
	 * @param debug Flag para imprimr en pantalla informacion extra.
	 * @throws Exception
	 */
	static void parseTSV(String fileName, utils.E_SamplingMethod sampling, boolean debug) throws Exception {
		TreeSet<String> dic = null; // diccionario de palabras posibles
		double[][] classes = null;
		double[][] data = null;
		
		Object[] dataNClassesNdic = loadTextData(data, classes, dic, fileName);
		data = (double[][]) dataNClassesNdic[0];
		classes = (double[][]) dataNClassesNdic[1];
		dic = (TreeSet<String>) dataNClassesNdic[2];
		
		dataNClassesNdic = utils.sampleData(data, classes, sampling);
		data = (double[][]) dataNClassesNdic[0];
		classes = (double[][]) dataNClassesNdic[1];
		
		// Inicio de red neuronal
		NeuralNetwork net = new NeuralNetwork(0.1);
		net.newInputLayer(dic.size(), dic.size());
		net.newHiddenLayer(15);
		net.newHiddenLayer(1);
		net.closeNetwork();

		// 50% datos de entrenamiento, 50% datos de prueba
		if (data.length != classes.length) {
			throw new Exception("parseTSV :: data size != classes size.");
		}
		net.train(data,
				classes, 
				20, 
				"SPAM");

		double threshold = 0.4;
		boolean verbose = true;
		HashMap<String, Double> metricsData = utils.binaryMetrics(net,
				data, 
				classes,
				threshold,
				verbose);
	}

	/**
	 * Carga datos de texto, le elimina stop-words, hace stemming. Codifica el texto en 'bag of words' (matriz de N x sizeOf(diccionario)), y le hace TF-IDF.
	 * Ademas, guarda resultado en archivo temporal para evitar hacer preprocesamiento de nuevo; para luego poder cargarlo.
	 * @param data Arreglo vacio donde quedara el 'bag of words' post-procesado.
	 * @param classes Arreglo vacio donde quedaran las clases (binarias), luego de parsear el String de la clase raw.
	 * @param dic TreeSet donde se almacenara el diccionario de palabras posibles.
	 * @param fileName Nombre del archivo donde estan los datos.
	 * @throws Exception
	 */
	static Object[] loadTextData(double[][] data, double[][] classes, TreeSet<String> dic, String fileName) throws Exception {
		List<String> rawClasses = new ArrayList<>(); // cara rawClass es un string
		List<List<String>> rawTexts = new ArrayList<>(); // cada sms como una lista de terminos
		try {
			ObjectInputStream inData = new ObjectInputStream(new FileInputStream(TFIDF_TEMPFILE_NAME + "_data"));
			ObjectInputStream inClasses = new ObjectInputStream(new FileInputStream(TFIDF_TEMPFILE_NAME + "_classes"));
			ObjectInputStream inDic = new ObjectInputStream(new FileInputStream(TFIDF_TEMPFILE_NAME + "_dic"));
			classes = (double[][]) inClasses.readObject();
			data = (double[][]) inData.readObject();
			dic = (TreeSet<String>) inDic.readObject();
			inClasses.close();
			inDic.close();
			inData.close();
		} catch (FileNotFoundException e) {
			// parse
			BufferedReader in = new BufferedReader(new FileReader(fileName));
			String line = in.readLine();
			while (line != null) {
				// parsear lineas del archivo
				String[] parsed = line.split("\\t", -1); // archivo 'tab separated'

				if (DEBUG) {
					System.out.println("raw_class: " + parsed[0] + " text: " + parsed[1]);
				}

				StringTokenizer st = new StringTokenizer(parsed[1], " "); // separar x palabras
				List<String> tokens = new ArrayList<>();
				while (st.hasMoreTokens()) {
					// iterar sobre terms de la linea
					String term = st.nextToken();
					// sacar stopwrods, eliminar caracteres especiales y hacer stemming
					term = term.replaceAll("[^a-zA-Z0-9]+", ""); // borrar caracteres especiales
					if (!Stopwords.isStopword(term)) {
						term = Stopwords.stemString(term);
						term = term.toLowerCase();
						tokens.add(term);
					}
				}
				rawTexts.add(tokens); // se agrega lista de terms a los textos
				rawClasses.add(parsed[0]); // se agrega a lista de clases
				line = in.readLine();
			}
			if (rawClasses.size() != rawTexts.size()) {
				throw new Exception("parseTSV :: Classes and Text have different lengths.");
			}

			// transformar rawClasses a numero binario
			classes = new double[rawClasses.size()][];
			int index = 0;
			for (String rawClass : rawClasses) {
				if (rawClass.equals("spam")) {
					classes[index++] = new double[] { 1 };
				} else {
					classes[index++] = new double[] { 0 };
				}
			}
			dic = new TreeSet<>(); // diccionario de palabras presentes en textos
			for (List<String> rawText : rawTexts) {
				for (String term : rawText) {
					dic.add(term);
				}
			}

			System.out.println("parseTSV :: Starting TF-IDF calculation for all texts.");
			TFIDF calculator = new TFIDF();
			data = new double[rawTexts.size()][dic.size()]; // bag of words
			/**
			 * Sistema de muestra de avanze. Muestra 10% cada 10% del proceso de TF_IDF
			 * 
			 */
			int step = rawTexts.size() / 10;
			int stepCounter = 0;
			/**/
			int dataIndex = 0;
			for (List<String> rawText : rawTexts) {
				// por cada sms
				int termIndex = 0;
				for (String term : dic) {
					// por cada elemento del diccionario posible
					if (rawText.contains(term)) {
						data[dataIndex][termIndex] = calculator.tfIdf(rawText, rawTexts, term);
					}
					termIndex++;
				}
				/**/
				if (stepCounter++ >= step) {
					stepCounter = 0;
					System.out.print("10%");
				}
				/**/
				dataIndex++;
			}
			// tengo X=data[][dic.len] y Y=classes[][1]

			// fill in your array here
			// data = utils.normalizeAny(data);
			ObjectOutputStream outData = new ObjectOutputStream(new FileOutputStream(TFIDF_TEMPFILE_NAME + "_data"));
			ObjectOutputStream outClasses = new ObjectOutputStream(
					new FileOutputStream(TFIDF_TEMPFILE_NAME + "_classes"));
			ObjectOutputStream outDic = new ObjectOutputStream(new FileOutputStream(TFIDF_TEMPFILE_NAME + "_dic"));

			outData.writeObject(data);
			outClasses.writeObject(classes);
			outDic.writeObject(dic);
			outData.close();
			outClasses.close();
		}
		return new Object[] {data, classes, dic};
	}

}

/********************************
Name: Christian Johnson
TOPIC: Naive Bayes
Date: July 10, 2023
********************************/

import java.util.*;
import java.io.*;

public class SimpleClassifier {
	static HashMap<String, Double> classFreq = new HashMap<String, Double>();

	// Make HashMap for each feature for each class
	static HashMap<String, Double> x1Pos = new HashMap<String, Double>();
	static HashMap<String, Double> x1Neg = new HashMap<String, Double>();
	static HashMap<String, Double> x2Pos = new HashMap<String, Double>();
	static HashMap<String, Double> x2Neg = new HashMap<String, Double>();
	static HashMap<String, Double> x3Pos = new HashMap<String, Double>();
	static HashMap<String, Double> x3Neg = new HashMap<String, Double>();
	// ArrayList for continuous data
	static ArrayList<Double> x4Pos = new ArrayList<Double>();
	static ArrayList<Double> x4Neg = new ArrayList<Double>();
	static ArrayList<Double> x5Pos = new ArrayList<Double>();
	static ArrayList<Double> x5Neg = new ArrayList<Double>();

	public static void main(String args[]) throws IOException {
		System.out.println("********************************************************************************");
		System.out.println("Naive Bayes Algorithm");
		System.out.println("Name:        Christian Johnson");
		System.out.println("Synax:       java SimpleClassifier " + args[0] + " " + args[1] + " " + args[2]); 
		System.out.println("********************************************************************************");
		System.out.println();
		System.out.println();
		train(args[0]); 
		test(args[1]);
		predict(args[2]);
	}

	public static void train(String filename) throws IOException {
		// get distinct classes
		HashSet<String> classes = new HashSet<String>();
		
		int featureNum = 0;

		BufferedReader br = new BufferedReader(new FileReader(filename));

		// loop through data to get distinct classes
		String line = br.readLine(); // skip first line which is col names
		while ((line = br.readLine()) != null) {
			String[] cols = line.split(",");
			String y = cols[cols.length - 1];
			classes.add(y);
			featureNum = cols.length - 1;

			if (classFreq.get(y) == null) {
				classFreq.put(y, 1.0);
			} else {
				classFreq.put(y, classFreq.get(y) + 1);
			}
		}
		

		br.close();

		System.out.printf("Training Phase: %s \n", filename);
		System.out.println("--------------------------------------------------------------");
		System.out.printf("    => Number of Entries (n): %10d \n", getTotal(classFreq));
		System.out.printf("    => Number of Features (p): %9d \n", featureNum);
		System.out.printf("    => Number of Distinct Classes (y): %1d \n", classes.size());

		System.out.println();
		System.out.println();

		// loop through again to make hash maps of frequencies of a feature givin its
		// class
		br = new BufferedReader(new FileReader(filename));

		line = br.readLine(); // skip first line

		// make dynamic later
		while ((line = br.readLine()) != null) {
			String[] cols = line.split(",");
			String y = cols[cols.length - 1];

			// if class is negative
			if (Integer.parseInt(y) == 0) {

				// add to feature 1 where class = 0
				if (x1Neg.get(cols[0]) == null) {
					x1Neg.put(cols[0], 1.0);
				} else {
					x1Neg.put(cols[0], x1Neg.get(cols[0]) + 1);
				}

				// add to feature 2 where class = 0
				if (x2Neg.get(cols[1]) == null) {
					x2Neg.put(cols[1], 1.0);

				} else {
					x2Neg.put(cols[1], x2Neg.get(cols[1]) + 1);

				}

				// add to feature 3 where class = 0
				if (x3Neg.get(cols[2]) == null) {
					x3Neg.put(cols[2], 1.0);

				} else {
					x3Neg.put(cols[2], x3Neg.get(cols[2]) + 1);

				}
				
				// add to feature 4 where class = 0
					x4Neg.add(Double.parseDouble(cols[3]));


				// add to feature 5 where class = 0
					x5Neg.add(Double.parseDouble(cols[4]));
				

			}

			// if class is positive
			if (Integer.parseInt(y) == 1) {

				// add to feature 1 where class = 1
				if (x1Pos.get(cols[0]) == null) {
					x1Pos.put(cols[0], 1.0);
				} else {
					x1Pos.put(cols[0], x1Pos.get(cols[0]) + 1);
				}

				// add to feature 2 where class = 1
				if (x2Pos.get(cols[1]) == null) {
					x2Pos.put(cols[1], 1.0);
				} else {
					x2Pos.put(cols[1], x2Pos.get(cols[1]) + 1);
				}

				// add to feature 3 where class = 1
				if (x3Pos.get(cols[2]) == null) {
					x3Pos.put(cols[2], 1.0);
				} else {
					x3Pos.put(cols[2], x3Pos.get(cols[2]) + 1);
				}
				
				// add to feature 4 where class = 0
				x4Pos.add(Double.parseDouble(cols[3]));


				// add to feature 5 where class = 0
				x5Pos.add(Double.parseDouble(cols[4]));

			}

		}

		br.close();

	}
	
	public static void test(String filename) throws IOException {
		int total = 0;
		int correct = 0;
		
		BufferedReader br = new BufferedReader(new FileReader(filename));

		String line = br.readLine();
		
		System.out.printf("Testing Phase: \n");
		System.out.println("--------------------------------------------------------------");

		System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%n", "F1", "F2", "F3", "F4", "F5",
				"CLASS", "PREDICT", "PROB", "RESULT");
		
		System.out.printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%n", "---", "---", "---", "-------", "-------",
				"-------", "-------", "-------", "-------");

		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");

			String f1 = tokens[0];
			String f2 = tokens[1];
			String f3 = tokens[2];
			double f4 = Double.parseDouble(tokens[3]);
			double f5 = Double.parseDouble(tokens[4]);
			String actualClass = tokens[5];
			String predictedClass = classify(f1, f2, f3, f4, f5);
			double prob = getHighestProb(f1, f2, f3, f4, f5) * 100;
			

			boolean isCorrect = actualClass.equals(predictedClass);

			if (isCorrect) {
				correct++;
			}

			total++;

			System.out.printf("%-10s %-10s %-10s %-10.3f %-10.3f %-10s %-10s %-10.1f %-10s%n", f1, f2, f3, f4,f5, actualClass, predictedClass, prob, isCorrect ? "CORRECT" : "INCORRECT");

		}

		double acc = (correct / (double) total) * 100;

		System.out.println();
		System.out.printf("Total Accuracy: %d correct / %d total = %.2f%% Accuracy%n", correct, total, acc);
		System.out.printf("    => Number of Entries (n): %10d \n", total);
		

		br.close();
	}
	
	public static void predict(String filename) throws IOException {
		System.out.println();
		System.out.printf("Prediction Phase: \n");
		System.out.println("--------------------------------------------------------------");
		System.out.printf("    %-10s %-10s %-10s %-10s %-10s %-10s %-10s%n", "F1", "F2", "F3", "F4", "F5", "PREDICT", "PROB");
		
		System.out.printf("    %-10s %-10s %-10s %-10s %-10s %-10s %-10s%n", "---", "---", "---", "-------", "-------",
				"-------", "-------");
		
		BufferedReader br = new BufferedReader(new FileReader(filename));

		String line;
		
		int total = 0;
		
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");

			String f1 = tokens[0];
			String f2 = tokens[1];
			String f3 = tokens[2];
			double f4 = Double.parseDouble(tokens[3]);
			double f5 = Double.parseDouble(tokens[4]);
			String predictedClass = classify(f1, f2, f3, f4, f5);
			double prob = getHighestProb(f1, f2, f3, f4, f5) * 100;


			total++;

			System.out.printf("    %-10s %-10s %-10s %-10.3f %-10.3f %-10s %-10.1f %n", f1, f2, f3, f4,f5, predictedClass, prob);
		}
		System.out.println();
		System.out.printf("    => Number of Entries (n): %10d \n", total);
		br.close();
	}

	public static String classify(String f1, String f2, String f3, double f4, double f5) throws IOException {
		double numeratorPos = getProbOfClass("1") * (getProbOfFeature1GivenClassPos(f1)
				* getProbOfFeature2GivenClassPos(f2) * getProbOfFeature3GivenClassPos(f3) * getProbOfFeature4GivenClassPos(f4) * getProbOfFeature5GivenClassPos(f5));
		double numeratorNeg = getProbOfClass("0") * (getProbOfFeature1GivenClassNeg(f1)
				* getProbOfFeature2GivenClassNeg(f2) * getProbOfFeature3GivenClassNeg(f3) * getProbOfFeature4GivenClassNeg(f4) * getProbOfFeature5GivenClassNeg(f5));
		double denom = (getProbOfClass("0") * (getProbOfFeature1GivenClassNeg(f1) * getProbOfFeature2GivenClassNeg(f2)
				* getProbOfFeature3GivenClassNeg(f3) * getProbOfFeature4GivenClassNeg(f4) * getProbOfFeature5GivenClassNeg(f5)))
				+ (getProbOfClass("1") * (getProbOfFeature1GivenClassPos(f1) * getProbOfFeature2GivenClassPos(f2)
						* getProbOfFeature3GivenClassPos(f3) * getProbOfFeature4GivenClassPos(f4) * getProbOfFeature5GivenClassPos(f5)));
		double cPos = numeratorPos / denom;
		double cNeg = numeratorNeg / denom;

		String choice = "";
		if (cPos > cNeg) {
			choice = "1";
		} else {
			choice = "0";
		}

		return choice;
	}

	public static double getHighestProb(String f1, String f2, String f3, double f4, double f5) throws IOException {
		double numeratorPos = getProbOfClass("1") * (getProbOfFeature1GivenClassPos(f1)
				* getProbOfFeature2GivenClassPos(f2) * getProbOfFeature3GivenClassPos(f3) * getProbOfFeature4GivenClassPos(f4) * getProbOfFeature5GivenClassPos(f5));
		double numeratorNeg = getProbOfClass("0") * (getProbOfFeature1GivenClassNeg(f1)
				* getProbOfFeature2GivenClassNeg(f2) * getProbOfFeature3GivenClassNeg(f3) * getProbOfFeature4GivenClassNeg(f4) * getProbOfFeature5GivenClassNeg(f5));
		double denom = (getProbOfClass("0") * (getProbOfFeature1GivenClassNeg(f1) * getProbOfFeature2GivenClassNeg(f2)
				* getProbOfFeature3GivenClassNeg(f3) * getProbOfFeature4GivenClassNeg(f4) * getProbOfFeature5GivenClassNeg(f5)))
				+ (getProbOfClass("1") * (getProbOfFeature1GivenClassPos(f1) * getProbOfFeature2GivenClassPos(f2)
						* getProbOfFeature3GivenClassPos(f3) * getProbOfFeature4GivenClassPos(f4) * getProbOfFeature5GivenClassPos(f5)));
		double cPos = numeratorPos / denom;
		double cNeg = numeratorNeg / denom;

		if (cPos > cNeg) {
			return cPos;
		} else {
			return cNeg;
		}
	}

	public static double getProbOfClass(String str) {
		double prob = classFreq.get(str) / getTotal(classFreq);

		return prob;
	}

	public static double getProbOfFeature1GivenClassNeg(String ft) {
		double prob = x1Neg.get(ft) / getTotal(x1Neg);

		return prob;
	}

	public static double getProbOfFeature1GivenClassPos(String ft) {
		double prob = x1Pos.get(ft) / getTotal(x1Pos);

		return prob;
	}

	public static double getProbOfFeature2GivenClassNeg(String ft) {
		double prob = x2Neg.get(ft) / getTotal(x2Neg);

		return prob;
	}

	public static double getProbOfFeature2GivenClassPos(String ft) {
		double prob = x2Pos.get(ft) / getTotal(x2Pos);

		return prob;
	}

	public static double getProbOfFeature3GivenClassNeg(String ft) {
		double prob = x3Neg.get(ft) / getTotal(x3Neg);

		return prob;
	}

	public static double getProbOfFeature3GivenClassPos(String ft) {
		double prob = x3Pos.get(ft) / getTotal(x3Pos);

		return prob;
	}
	public static double getProbOfFeature4GivenClassNeg(double ft) throws IOException {
		double prob = getPDF(x4Neg, ft);
			
		return prob;
	}

	public static double getProbOfFeature4GivenClassPos(double ft) throws IOException {
		double prob = getPDF(x4Pos, ft);

		return prob;
	}
	public static double getProbOfFeature5GivenClassNeg(double ft) throws IOException {
		double prob = getPDF(x5Neg, ft);

		return prob;
	}

	public static double getProbOfFeature5GivenClassPos(double ft) throws IOException {
		double prob = getPDF(x5Pos, ft);

		return prob;
	}

	public static int getTotal(HashMap<String, Double> hm) {
		int total = 0;

		for (String key : hm.keySet()) {
			total += hm.get(key);
		}

		return total;
	}

	// methods to calculate mean, variance, standard deviation, pdf 
	public static double[] listIntoVec(ArrayList<Double> list) throws IOException {

		int numRows = 0;

		// get matrix size first	
		for(int i = 0; i < list.size(); i++) {
			numRows++;
		}

		// now make vec with number of rows
		double[] vec = new double[numRows];

		for(int i = 0; i < list.size(); i++) {
			vec[i] = list.get(i);
		}


		return vec;
	}

	public static double getAverage(double[] x) {
		double total = 0;

		for (int row = 0; row < x.length; row++) {
			total += x[row];
		}

		return total / x.length;
	}


	public static double getVariance(double[] x) {
		double avg = getAverage(x);
		
		double total = 0;
		
		for(int row = 0; row < x.length; row++) {
			total += Math.pow((x[row] - avg), 2);
		}
		
		return total / x.length - 1;
	}

	public static double getStandardDeviation(double x) {	
		return Math.sqrt(x);
	}
	
	public static double getPDF(ArrayList<Double> list, double x ) throws IOException {
		double[] feature = listIntoVec(list);
		
		double avg = getAverage(feature);
		double variance = getVariance(feature);
		double sd = getStandardDeviation(variance);
		
		
		double exponent = (-1/2) * Math.pow(((x - avg) / sd), 2);
		
		double pdf = (1 / (sd * Math.sqrt(2*Math.PI))) * Math.pow(Math.exp(1), exponent);
		
		return pdf;
	}

}

//A NN model for behaviour prediction in a game

public class BPMain {

//loading the training data
public static double	TrainingSet[][] = {  
	    //friends health  enemy    range   chase   flock   evade
			{0,		1,		0,		0.2,	0.9,	0.1,	0.1},
			{0,		1,		1,		0.2,	0.9,	0.1,	0.1},
			{0,		1,		0,		0.8,	0.1,	0.1,	0.1},
			{0.1,	0.5,	0,		0.2,	0.9,	0.1,	0.1},
			{0,		0.25,	1,		0.5,	0.1,	0.9,	0.1},
			{0,		0.2,	1,		0.2,	0.1,	0.1,	0.9},
			{0.3,	0.2,	0,		0.2,	0.9,	0.1,	0.1},
			{0,		0.2,	0,		0.3,	0.1,	0.9,	0.1},
			{0,		1,		0,		0.2,	0.1,	0.9,	0.1},
			{0,		1,		1,		0.6,	0.1,	0.1,	0.1},
			{0,		1,		0,		0.8,	0.1,	0.9,	0.1},
			{0.1,	0.2,	0,		0.2,	0.1,	0.1,	0.9},
			{0,		0.25,	1,		0.5,	0.1,	0.1,	0.9},
			{0,		0.6,	0,		0.2,	0.1,	0.1,	0.9}
			};


private static NeuralNetwork TheBrain = new NeuralNetwork();

//training the neural networks using the above training data
public static void TrainTheBrain()
{
	int		i;
	double	error = 1;
	int		c = 0;
    
	System.out.println("************Before training*******************");
	
	//save the initial weights in a file before the training
	TheBrain.DumpData();

	//training loops
	while((error > 0.05) && (c<50000))
	{
		error = 0;
		c++; //iteration number
		for(i=0; i<14; i++)
		{
			//set the inputs for each training instance
			TheBrain.SetInput(0, TrainingSet[i][0]);
			TheBrain.SetInput(1, TrainingSet[i][1]);
			TheBrain.SetInput(2, TrainingSet[i][2]);
			TheBrain.SetInput(3, TrainingSet[i][3]);

			//set the desired outputs for each training instance
			TheBrain.SetDesiredOutput(0, TrainingSet[i][4]);
			TheBrain.SetDesiredOutput(1, TrainingSet[i][5]);
			TheBrain.SetDesiredOutput(2, TrainingSet[i][6]);

			//calculate the real output from the NN model for each training instance
			TheBrain.FeedForward();
			
			//identify the error for each instance
			error += TheBrain.CalculateError();
			
			//adjust the weights using the errors identified between the real and the ideal outputs
			TheBrain.BackPropagate();

		}
		//calculate the mean error for the all the training data
		error = error / 14.0f;
	}

	//c = c * 1;
	System.out.println("************After training*******************");
	//save the trained weights in a file after the training is complete.
	TheBrain.DumpData();


}

//testing the trained neural network
public static void TestTheBrain()
{
	
	System.out.println("Output results");
	for (int i=0; i<14; i++)
	{
		
	//set up the inputs for each test instance
    TheBrain.SetInput(0, TrainingSet[i][0]);
    TheBrain.SetInput(1, TrainingSet[i][1]);
    TheBrain.SetInput(2, TrainingSet[i][2]);
    TheBrain.SetInput(3, TrainingSet[i][3]);

    //calculate the real output for each test instance
    TheBrain.FeedForward();
    
    System.out.println("\n");
	System.out.println("--------------------------------------------------------");
	System.out.print((i+1) + " ");

	//identify the action with the highest probability among the three predicted behaviours for each test instance
	double max = -1000.0;
	int index = -1000;
	for(int j=0; j<3; j++)
	{
		System.out.print(TheBrain.GetOutput(j) + "; ");
		if (max < TheBrain.GetOutput(j))
			{ 
			   max = TheBrain.GetOutput(j);
			   index = j;
			}		
	}
	
	//System.out.print("index : " + index);
	//output the predicted most likely behaviour on the screen
	if (index == 0)
		System.out.print(" chase : " + TheBrain.GetOutput(index) + "; ");
	else if (index == 1)
	    System.out.print(" flock : " + TheBrain.GetOutput(index) + "; ");
	else if (index == 2)
		System.out.print(" evade : " + TheBrain.GetOutput(index) + "; ");
	
	}
	
	System.out.println("\n");
}



public static void main(String args[])
{
	
	TheBrain.Initialize(4, 3, 3); //initialise the structure of the neural networks
	TheBrain.SetLearningRate(0.2); //setting the learning rate
	TheBrain.SetMomentum(true, 0.9); //setting up other parameters
	TrainTheBrain(); //train the neural networks
    TestTheBrain();  //test the neural networks

}

}



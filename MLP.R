#This is a simple example solution of the lab exercise in Week 3 Attack of Flee game. Please use this with the lab session slides and lab exercise brief.
sigmoid = function(x){
	1/(1+exp(-x))
}

#Training data set
TrainingSet =   
	    #friends health  enemy	range   chase   flock		evade
rbind(	c(0,		1,		0,	0.2,		0.9,		0.1,		0.1),
		c(0,		1,		1,	0.2,		0.9,		0.1,		0.1),
		c(0,		1,		0,	0.8,		0.1,		0.1,		0.1),
		c(0.1,		0.5,	0,	0.2,		0.9,		0.1,		0.1),
		c(0,		0.25,	1,	0.5,		0.1,		0.9,		0.1),
		c(0,		0.2,	1,	0.2,		0.1,		0.1,		0.9),
		c(0.3,		0.2,	0,	0.2,		0.9,		0.1,		0.1),
		c(0,		0.2,	0,	0.3,		0.1,		0.9,		0.1),
		c(0,		1,		0,	0.2,		0.1,		0.9,		0.1),
		c(0,		1,		1,	0.6,		0.1,		0.1,		0.1),
		c(0,		1,		0,	0.8,		0.1,		0.9,		0.1),
		c(0.1,		0.2,	0,	0.2,		0.1,		0.1,		0.9),
		c(0,		0.25,	1,	0.5,		0.1,		0.1,		0.9),
		c(0,		0.6,	0,	0.2,		0.1,		0.1,		0.9))
colnames(TrainingSet)=c("friends", "health", "enemy", "range", "chase", "flock", "evade")

## 
## definitions:
INPUT.UNIT.NUM =4
HIDDEN.UNIT.NUM=3
OUTPUT.UNIT.NUM=3
LearningRate = 0.2

## weights - layer1: from input to hidden
WGT.LYR.IH=matrix(rnorm(HIDDEN.UNIT.NUM*(INPUT.UNIT.NUM+1)),  nrow=HIDDEN.UNIT.NUM)

## weights - layer2: from hidden to output
WGT.LYR.HO=matrix(rnorm(OUTPUT.UNIT.NUM*(HIDDEN.UNIT.NUM+1)), nrow=OUTPUT.UNIT.NUM)
mse.limit=0.05
max.iter=1000
iter=0
mse.iter=1

while(mse.iter > mse.limit & iter < max.iter){
	mse.iter=0
	iter=iter+1
	for(ind in 1:nrow(TrainingSet)){

		########################################################################################
		## FEEDFORWARD:
		## from input to hidden layer:
		hidden.nodes=c()
		for(hid in 1:HIDDEN.UNIT.NUM) ## in the end, do not forget adding the weight of the bias, i.e. WGT.LYR.IH[hid, (INPUT.UNIT.NUM+1)]
			hidden.nodes[hid] = sigmoid( TrainingSet[ind, 1:INPUT.UNIT.NUM] %*% WGT.LYR.IH[hid, 1:INPUT.UNIT.NUM] + WGT.LYR.IH[hid, (INPUT.UNIT.NUM+1)] )
		hidden.nodes[HIDDEN.UNIT.NUM+1]=1.0 ## value of bias in the hidden layer 1.0
		
		## from hidden to output layer:
		output.nodes=c()
		for(out in 1:OUTPUT.UNIT.NUM) ## in the end, do not forget adding the weight of the bias, i.e. WGT.LYR.HO[out, (HIDDEN.UNIT.NUM+1)]
			output.nodes[out] = sigmoid( hidden.nodes %*% WGT.LYR.HO[out, ] )
		########################################################################################
		
		########################################################################################
		## calculate the MSE of this data instance and add it to the current value:
		mse.iter = mse.iter + sum((output.nodes-TrainingSet[ind, (INPUT.UNIT.NUM+1):ncol(TrainingSet)]) *
			 (output.nodes-TrainingSet[ind, (INPUT.UNIT.NUM+1):ncol(TrainingSet)])) / OUTPUT.UNIT.NUM
		########################################################################################

		########################################################################################
		## BACKPROPOGATE ERRORS:
		## The errors based on the equation in Slide 13 of lab session slides:
		## errors from the output layer to the hidden layer
		error.output=c()
		for(out in 1:OUTPUT.UNIT.NUM) ## error.out=real_value-predicted_value
			error.output[out] = (TrainingSet[ind, INPUT.UNIT.NUM+out] - output.nodes[out]) * 
				output.nodes[out] * (1-output.nodes[out]) ## derivative_of_sigmoid= sigmoid*(1-sigmoid)
		##
		## errors from hidden to input layer:
		error.hidden=c()
		for(hid in 1:HIDDEN.UNIT.NUM) {
			sum=0
			for(out in 1:OUTPUT.UNIT.NUM)
				sum=sum + error.output[out] * WGT.LYR.HO[out, hid]
			error.hidden[hid] = sum * hidden.nodes[hid] * (1-hidden.nodes[hid])
		}
		########################################################################################

		########################################################################################
		## UPDATE THE WEIGHT based on the equation in Slide 14 of lab session slides
		## from output to hidden layer:
		for(out in 1:OUTPUT.UNIT.NUM)
			for(hid in 1:(HIDDEN.UNIT.NUM+1) )  ## do not forget the weight of the bias (1.0) in hidden layer
				WGT.LYR.HO[out, hid] = WGT.LYR.HO[out, hid] + LearningRate * error.output[out] * hidden.nodes[hid]
		##
		## from hidden to input layer:
		for(hid in 1:HIDDEN.UNIT.NUM)
			for(inp in 1:(INPUT.UNIT.NUM +1) )  ## do not forget the weight of the bias (1.0) in input layer
				WGT.LYR.IH[hid, inp] = WGT.LYR.IH[hid, inp]  + LearningRate * error.hidden[hid] * TrainingSet[ind, inp]
		########################################################################################
	}
	
	#calculate the mean error for the all the training data
	mse.iter = mse.iter / nrow(TrainingSet);
	print(paste0("In the ", iter, "-th iteration, MSE is: ", mse.iter))
}


### AFTER THE TRAINING, OUTPUTS FOR THE TRAINING INSTANCES ARE AS FOLLOWS:
## one may use the 'test instances' to predict the behaviour of creature 
for(ind in 1:nrow(TrainingSet)){
	########################################################################################
	## FEEDFORWARD:
	## from input to hidden layer:
	hidden.nodes=c()
	for(hid in 1:HIDDEN.UNIT.NUM) ## in the end, do not forget adding the weight of the bias, i.e. WGT.LYR.IH[hid, (INPUT.UNIT.NUM+1)]
		hidden.nodes[hid] = sigmoid( TrainingSet[ind, 1:INPUT.UNIT.NUM] %*% WGT.LYR.IH[hid, 1:INPUT.UNIT.NUM] + WGT.LYR.IH[hid, (INPUT.UNIT.NUM+1)] )
	hidden.nodes[HIDDEN.UNIT.NUM+1]=1.0 ## value of bias in the hidden layer 1.0
		
	## from hidden to output layer:
	output.nodes=c()
	for(out in 1:OUTPUT.UNIT.NUM) ## in the end, do not forget adding the weight of the bias, i.e. WGT.LYR.HO[out, (HIDDEN.UNIT.NUM+1)]
		output.nodes[out] = sigmoid( hidden.nodes %*% WGT.LYR.HO[out, ] )
	output.nodes = round(output.nodes, 2) 	
	print(paste0("For the ", ind, "-th data instance, prediction of chase is: ", output.nodes[1], " ; flock is: ",  output.nodes[2], " ; evade is: ", output.nodes[3]))
	print(paste0("For the ", ind, "-th data instance, real value of chase is: ", TrainingSet[ind,5], " ; flock is: ", TrainingSet[ind,6], " ; evade is: ", TrainingSet[ind,7]))
	
}




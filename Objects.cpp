#include "Objects.h"

Objects::Objects(void) {

	//set values for default constructor
	setType("null");
	setColour(Scalar(0,0,0));
}

Objects::~Objects(void) {}


Objects::Objects(string name) {

	setType(name);
	
	if(name=="person") {

		//TODO: use "calibration mode" to find HSV min
		//and HSV max values

		setHSVMin(Scalar(0,0,0));
		setHSVMax(Scalar(255,255,255));

		//BGR value for Green:
		setColour(Scalar(0,255,0));

	}
}

int Objects::getXPos() {

	return Objects::xPos;
}

int Objects::getYPos() {

	return Objects::yPos;
}

void Objects::setXPos(int x) {

	Objects::xPos = x;
}

void Objects::setYPos(int y) {

	Objects::yPos = y;
}

Scalar Objects::getHSVMin() {

	return Objects::HSVMin;
}

Scalar Objects::getHSVMax() {

	return Objects::HSVMax;
}

void Objects::setHSVMax(Scalar max) {

	Objects::HSVMax = max;
}

void Objects::setHSVMin(Scalar min) {

	Objects::HSVMin = min;
}
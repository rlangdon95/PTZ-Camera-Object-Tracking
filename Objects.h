#pragma once

#include <vector>
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#pragma once
class Objects {

	public:
		Objects(void);
		~Objects(void);

		Objects(string name);

		int getXPos();
		int getYPos();
	
		void setXPos(int x);
		void setYPos(int y);

		Scalar getHSVMin();
		Scalar getHSVMax();

		void setHSVMin(Scalar x);
		void setHSVMax(Scalar y);

		string getType(){return type;}
		void setType(string t){type = t;}

		Scalar getColour() {
		
			return Colour;
		}

		void setColour(Scalar c) {

			Colour = c;
		}

	private:

		int xPos, yPos;
		string type;

		Scalar HSVMin, HSVMax;
		Scalar Colour;
};
#include "ros/ros.h"

void cudamain();
void cudamain1();

int main(int argc, char **argv)
{
	ros::init(argc, argv, "roscuda_basic");	  
	ros::NodeHandle n;
	ros::Rate loop_rate(1);

	while(ros::ok())
	{    
		cudamain1();
	    ros::spinOnce();
	    loop_rate.sleep();
	}
	return 0;
}

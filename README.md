# mlops_pos
A project that aims to create a well-organized ML pipeline for pose the estimation task.

## Relevance
The pose estimation problem is quite popular and very practical, and can be applied in various areas. From movie production and gaming to streaming, the topic was mainly inspired by the popularity of vtubers. vtubers are using machine learning to help them register their movements and depict them on the screen in some avatars. The quality of such translation is quite high today, and we can apply known solutions.

## Architecture and data
Many architectures can technically be reused in the pose estimation task (from R-CNN) but I consider using YOLO-v8 as a baseline and, depending on the result quality, searching for more solutions. 
There are a lot of open datasets for the pose estimation task, and some of them are very large ([this dataset](https://academictorrents.com/details/34f2197d360ac8453b33f50d09e452d504d30cbb) and [that one](http://human-pose.mpi-inf.mpg.de/#overview)). The mentioned datasets are going to be our baseline, and we might increase it from there. There are over 40 thousand labeled images in these datasets covering verious human poses in different sitiations (focusing on upper half of the body).

## Application
In my view, the process should look as follows: an image is fed into the pipeline, and the model processes it, resulting in the estimated pose. After that, we use one of the prepared avatars and map it onto the pose. Perfectly, this process could be animated, but I am not yet sure how difficult it's going to be. The avatar may be displayed on the original image instead of the person, but that would require size matching, which is complicated. Thus, I consider displaying the result on different static images. Generally speaking, this is a project to simulate a vtuber experience.

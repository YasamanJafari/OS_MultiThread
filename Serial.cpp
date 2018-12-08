#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <string>
#include <string.h>
#include <semaphore.h>
#include <pthread.h>

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder

#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

#define NUMBER_OF_INPUT_CELLS 784   ///< use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_HIDDEN_CELLS 256   ///< use 256 hidden cells in one hidden layer
#define NUMBER_OF_OUTPUT_CELLS 10   ///< use 10 output cells to model 10 digits (0-9)

#define MNIST_MAX_TESTING_IMAGES 10000                      ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28                                  ///< image width in pixel
#define MNIST_IMG_HEIGHT 28                                 ///< image height in pixel

using namespace std;

typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;
typedef struct Hidden_Node Hidden_Node;
typedef struct Output_Node Output_Node;
/**
 * @brief Data block defining a hidden cell
 */

struct Hidden_Node{
    double weights[28*28];
    double bias;
    double output;
};

/**
 * @brief Data block defining an output cell
 */

struct Output_Node{
    double weights[256];
    double bias;
    double output;
};
Hidden_Node hidden_nodes[NUMBER_OF_HIDDEN_CELLS];
Output_Node output_nodes[NUMBER_OF_OUTPUT_CELLS];

void* readImage(void*);
void* display(void*);
void* sum_result(void*);
void* process(void*);
void createInputThreadAndSemaphores();
void createHiddenThreadAndSemaphores();
void createOutputThreadAndSemaphores();
void createDisplayThreadAndSemaphores();

sem_t input_mutex, hidden_mutex, hidden_progress_mutex, sum_mutex, sum_progress_mutex, display_mutex, input_display_mutex, output_display_mutex;
sem_t inside_hidden_semaphores[8];
sem_t inside_output_semaphores[10];

/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_Image{
    uint8_t pixel[28*28];
};

/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

vector <pthread_t> threadIDs;
MNIST_Image img;
int hidden_thread_count = 8;

/**
 * @details Set cursor position to given coordinates in the terminal window
 */

void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}

/**
 * @details Clear terminal screen by printing an escape sequence
 */

void clearScreen(){
    printf("\e[1;1H\e[2J");
}

/**
 * @details Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 */

void displayImage(MNIST_Image *img, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");

    for (int y=0; y<MNIST_IMG_HEIGHT; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
        strcat(imgStr,"|");

        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
}

/**
 * @details Outputs a 28x28 text frame at a defined screen position
 */

void displayImageFrame(int row, int col){

    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------");

}

/**
 * @details Outputs reading progress while processing MNIST testing images
 */

void displayLoadingProgressTesting(int imgCount, int y, int x){

    float progress = (float)(imgCount+1)/(float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);

}

/**
 * @details Outputs image recognition progress and error count
 */

void displayProgress(int imgCount, int errCount, int y, int x){

    double successRate = 1 - ((double)errCount/(double)(imgCount+1));

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n",imgCount+1-errCount, errCount, successRate*100);


}

/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n){

    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

    return (b0 | b1 | b2 | b3);

}

/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){

    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){

    lfh->magicNumber =0;
    lfh->maxImages   =0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);

}

/**
 * @details Open MNIST image file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st IMAGE
 */

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}


/**
 * @details Open MNIST label file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st LABEL
 */

FILE *openMNISTLabelFile(char *fileName){

    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}

/**
 * @details Returns the next image in the given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile){

    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

/**
 * @details Returns the next label in the given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile){

    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

/**
 * @brief allocate weights and bias to respective hidden cells
 */

void allocateHiddenParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(HIDDEN_WEIGHTS_FILE);
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 28*28; ++i){
            in >> hidden_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> hidden_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

/**
 * @brief allocate weights and bias to respective output cells
 */

void allocateOutputParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 256; ++i){
            in >> output_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> output_nodes[bidx].bias;
        bidx++;
    }
    biases.close();
}

/**
 * @details The output prediction is derived by finding the maxmimum output value
 * and returning its index (=0-9 number) as the prediction.
 */

int getNNPrediction(){

    double maxOut = 0;
    int maxInd = 0;

    for (int i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){

        if (output_nodes[i].output > maxOut){
            maxOut = output_nodes[i].output;
            maxInd = i;
        }
    }

    return maxInd;
}

void* readImage(void*){
    FILE *imageFile; 
    imageFile = openMNISTImageFile((char*)MNIST_TESTING_SET_IMAGE_FILE_NAME);
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
            // cerr << endl << 1 << endl;
        for(int i = 0; i < hidden_thread_count; i++){
            sem_wait(&input_mutex);
        }
        sem_wait(&input_display_mutex);
        displayLoadingProgressTesting(imgCount,5,5);
        img = getImage(imageFile);

        for(int i = 0; i < hidden_thread_count; i++){
            sem_post(&hidden_mutex);
        }
        cerr << endl << 2 << endl;
        displayImage(&img, 8,6);
        sem_post(&output_display_mutex);
    }
    fclose(imageFile);

    return NULL; 
}

void* process(void* count){

    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        // cerr << endl << 3 << " " << *(int *)count << endl;
        sem_wait(&hidden_mutex);
        // cerr << "   @!#!@#! " <<  *(int *)count << endl;
        for(int i = 0; i < 10; i++)
            sem_wait(&inside_hidden_semaphores[*(int *)count]);
        for (int j = 0; j < (*(int *)count + 1) * 32; j++) {
            hidden_nodes[j].output = 0;
            for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
                hidden_nodes[j].output += img.pixel[z] * hidden_nodes[j].weights[z];
            }
            hidden_nodes[j].output += hidden_nodes[j].bias;
            hidden_nodes[j].output = (hidden_nodes[j].output >= 0) ?  hidden_nodes[j].output : 0;
        }
        for(int i = 0; i < 10; i++)
            sem_post(&inside_output_semaphores[i]);
        sem_post(&input_mutex);
        // cerr << endl <<"* " << *(int *)count << endl;
        // cerr << endl << 4 << endl;
    }
    return NULL; 
}

void* sum_result(void* count){
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
            // cerr << endl << 5 << endl;
        sem_wait(&sum_progress_mutex);
        for(int k = 0; k < hidden_thread_count; k++)
            sem_wait(&inside_output_semaphores[*(int*)count]);
        for (int i= 0; i < NUMBER_OF_OUTPUT_CELLS; i++){
            output_nodes[i].output = 0;
            for (int j = 0; j < NUMBER_OF_HIDDEN_CELLS; j++) {
                output_nodes[i].output += hidden_nodes[j].output * output_nodes[i].weights[j];
            }
            output_nodes[i].output += 1/(1+ exp(-1* output_nodes[i].output));
        }
        for(int k = 0; k < hidden_thread_count; k++)
            sem_post(&inside_hidden_semaphores[k]);
        sem_post(&display_mutex);
            // cerr << endl << 6 << endl;
    }
    return NULL; 
}

void* display(void*){
    int errCount = 0;

    // open MNIST files
    FILE *labelFile;
    labelFile = openMNISTLabelFile((char*)MNIST_TESTING_SET_LABEL_FILE_NAME);

    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        // cerr << endl << 7 << endl;
        for(int i = 0; i < 10; i++)
            sem_wait(&display_mutex);
        sem_wait(&output_display_mutex);
        MNIST_Label lbl = getLabel(labelFile);
        int predictedNum = getNNPrediction();
        if (predictedNum!=lbl) 
            errCount++;

        printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);

        displayProgress(imgCount, errCount, 5, 66);
        sem_post(&input_display_mutex);
        for(int k = 0; k < 10; k++)
            sem_post(&sum_progress_mutex);
        // cerr << endl << 8 << endl;
    }
    // Close file
    fclose(labelFile);
    return NULL;
}

/**
 * @details test the neural networks to obtain its accuracy when classifying
 * 10k images.
 */

void createInputThreadAndSemaphores(){
    pthread_t threadID;
    pthread_create(&threadID, NULL, readImage, NULL);
    threadIDs.push_back(threadID);
    sem_init(&input_mutex, 0, hidden_thread_count); 
    sem_init(&input_display_mutex, 0, 1);
}

void createHiddenThreadAndSemaphores(){
    pthread_t threadID;
    int* count = (int*)malloc(sizeof(int) * hidden_thread_count);
    for(int i = 0; i < hidden_thread_count; i++)
    {
        count[i] = i;
        pthread_create(&threadID, NULL, process, (void *)(count+i)); 
        threadIDs.push_back(threadID);  
        sem_init(&inside_hidden_semaphores[i], 0, 10);      
    }
    sem_init(&hidden_mutex, 0, 0);
}

void createOutputThreadAndSemaphores(){
    pthread_t threadID;
    int* count = (int*)malloc(sizeof(int) * 10);
    for(int i = 0; i < 10; i++)
    {
        count[i] = i;
        pthread_create(&threadID, NULL, sum_result, (void *)(count+i)); 
        threadIDs.push_back(threadID); 
        sem_init(&inside_output_semaphores[i], 0, 8);        
    }   
    sem_init(&sum_progress_mutex, 0, 10);
}

void createDisplayThreadAndSemaphores(){
    pthread_t threadID;
    pthread_create(&threadID, NULL, display, NULL);
    threadIDs.push_back(threadID);
    sem_init(&display_mutex, 0, 0);
    sem_init(&output_display_mutex, 0, 0);
}

void testNN(){
    // screen output for monitoring progress
    displayImageFrame(7,5);

    createInputThreadAndSemaphores();
    createHiddenThreadAndSemaphores();
    createOutputThreadAndSemaphores();
    createDisplayThreadAndSemaphores();

    for(int i = 0; i < threadIDs.size(); i++)
    {
        pthread_join(threadIDs[i], NULL); 
    }
}
int main(int argc, const char * argv[]) {

    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);

    // clear screen of terminal window
    clearScreen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");

    // alocating respective parameters to hidden and output layer cells
    allocateHiddenParameters();
    allocateOutputParameters();

    //test the neural network
    testNN();

    locateCursor(38, 5);

    // calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);

    return 0;
}

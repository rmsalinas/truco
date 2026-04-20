#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>
#include <filesystem>
#include <fstream>
#include "findtrucontour.h"
#include "bordertrackeromp.h"
#include <random>    // for std::mt19937 and std::random_device
int GlobalcontourSizeThres=1;
int testNTimesMain=4;
int testNTimes=5;
float scale=1;
int minThreads=1;
int maxThreads=20;
bool TEST_TRUCO=true;
bool TEST_TRUCO_DEV_SAME_AS_SUZUKI=false;


bool TEST_FINDCONTOUR=true;
bool TEST_FINDCONTOURRUNLINKS=true;
bool TEST_GARCIA=true;

std::string baseComparison="01::findContours";
bool calculateThreadLoadBalance=false;
//helper to  change the number of threads
struct CvNThreadScope{
    int nprev;
    CvNThreadScope(int n){
        nprev=cv::getNumThreads();
        cv::setNumThreads(n);
    }
    ~CvNThreadScope(){
        cv::setNumThreads(nprev);
    }
};

bool areEquivalent( const std::vector<std::vector<cv::Point>> &cont1,const std::vector<std::vector<cv::Point>> &cont2);
class Timer{
    int64 tstart,tend,bestTime=std::numeric_limits<int64>::max();
public:
    inline void start(){tstart=cv::getTickCount();        }
    void end(){tend=cv::getTickCount();bestTime= std::min(bestTime,tend-tstart);}
    double getBestTime(){return (bestTime)/cv::getTickFrequency()*1000.0;}
};

template<typename T>
struct maxinit{
    T val=std::numeric_limits<T>::max();
    T operator()(){return val;}
    T&operator=(T v){val=v;return val;};
};
std::map<std::string,maxinit<double>> bestTimes;
void tests( cv::Mat &imcolorin,std::map<std::string,maxinit<double>> &bestTimes);

std::string getCSVFileName(std::string csvUserFileName,double scale){
    std::string csvFileName;
    if(csvUserFileName.empty())
        csvFileName+="test_scale_"+std::to_string(scale)+".csv";
    else
        csvFileName=csvUserFileName;
    return csvFileName;
}

int main(int argc, char** argv)
{

    if(argc<2){
        std::cerr<<"Usage: "<<argv[0]<<" <image_folder> [-csv filename] [-scale val] [-binary] [-clean] [-balance] [-minsize val] [-no-METHOD] [-minthreads val] [-maxthreads val] [-testTRUCOasSuzuki]"<<std::endl;
        return -1;
    }

    bool firstTimeCSV=true;
    bool binary=false;
    bool singleImage=false;
    bool clean=false;
    std::string csvUserFileName;
    //parse optional args
    for(int i=2;i<argc;i++){
        std::string arg=argv[i];
        if(arg=="-scale" && i+1<argc){
            scale=std::stof(argv[i+1]);
            i++;
            std::cerr<<"Using scale "<<scale<<std::endl;
        }
        if(arg=="-binary"){
            binary=true;
            std::cerr<<"Not applying thresholding. Assuming binary image"<<std::endl;
        }
        if(arg=="-clean"){
            clean=true;
            std::cerr<<"Cleaning previous results"<<std::endl;
        }
        if(arg=="-balance")
            calculateThreadLoadBalance=true;
        if(arg=="-csv" && i+1<argc){
            csvUserFileName=argv[i+1];
            i++;
            std::cerr<<"Using csv file "<<csvUserFileName<<std::endl;
        }
        if(arg=="-minsize" && i+1<argc){
            GlobalcontourSizeThres=std::stoi(argv[i+1]);
            i++;
            std::cerr<<"Using min contour size "<<GlobalcontourSizeThres<<std::endl;
        }
        if(arg=="-no-findContours")
            TEST_FINDCONTOUR=false;
        if(arg=="-no-findContoursLinkRuns")
            TEST_FINDCONTOURRUNLINKS=false;
        if(arg=="-no-GarciaEtAl")
            TEST_GARCIA=false;
         if(arg=="-no-TRUCO")
            TEST_TRUCO=false;
         if(arg=="-minthreads" && i+1<argc){
            minThreads=std::stoi(argv[i+1]);
            i++;
            std::cerr<<"Using min threads "<<minThreads<<std::endl;
         }
         if(arg=="-maxthreads" && i+1<argc){
            maxThreads=std::stoi(argv[i+1]);
            i++;
            std::cerr<<"Using max threads "<<maxThreads<<std::endl;
         }

         if(arg=="-testTRUCOasSuzuki"){
            TEST_TRUCO_DEV_SAME_AS_SUZUKI=true;
            std::cerr<<"Testing findTRUContours results against Suzuki findContours."<<std::endl;
         }
    }
    ///////////////////////////////////////
    /// SINGLE IMAGE JUST FOR TESTING
    //if a file and jpg extension or png
    ///////////////////////////////////////
    std::filesystem::path argv1(argv[1]);
    if(std::filesystem::is_regular_file(argv1)  && (argv1.extension()==".jpg" || argv1.extension()==".png")){
        cv::Mat im,imcolorin=cv::imread(argv[1]);
        auto sizex=imcolorin.cols*scale;
        auto sizey=imcolorin.rows*scale;
        cv::resize(imcolorin,imcolorin,cv::Size(sizex,sizey));
        cv::cvtColor(imcolorin,im,cv::COLOR_BGR2GRAY);
        std::cerr<<"Size="<<im.cols<<" x "<<im.rows<<std::endl;
        cv::Mat imthres;
        if(!binary){
            //cv::Canny(im,imthres,50,200);
            cv::adaptiveThreshold(im,imthres,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY_INV,11,5);
        }
        else
            cv::threshold(im,imthres,128,255,cv::THRESH_BINARY);

        cv::imwrite("imthres.jpg",imthres);
        for(int i=0;i<testNTimesMain;i++){
            tests(imthres,bestTimes);
            double base;
            if(bestTimes.contains(baseComparison))
                base=bestTimes[baseComparison].val;
            else base=bestTimes.begin()->second.val;
            std::cerr<<"---- Speedup table ----"<<std::endl;
            for(auto&p:bestTimes){
                std::cerr<<"    "<<p.first<<" : "<<p.second.val<<" ms, speedup "<<base/p.second.val<<std::endl;
            }
        }
        return 0;
    }


    if(clean){
        //add scale
        if( std::filesystem::exists( getCSVFileName(csvUserFileName,scale))){
            std::filesystem::remove(getCSVFileName(csvUserFileName,scale));
            std::cerr<<"Removed "<<getCSVFileName(csvUserFileName,scale)<<std::endl;
        }
    }

    //add scale

    //read test.csv if exists
    std::set<std::string> filenames;
    if( std::filesystem::exists(getCSVFileName(csvUserFileName,scale))){
        std::ifstream inCSv(getCSVFileName(csvUserFileName,scale));
        std::string line;
        //read header
        std::getline(inCSv,line);
        if(line.size()>5){
            firstTimeCSV=false;
        }
        while(std::getline(inCSv,line)){
            std::istringstream ss(line);
            std::string filename;
            std::getline(ss,filename,',');
            filenames.insert(filename);
            std::cerr<<"Already processed "<<filename<<std::endl;
        }
    }

    std::ofstream outCSv(getCSVFileName(csvUserFileName,scale),std::ios::app);
    std::vector<std::filesystem::path> images;
    for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
        images.push_back(entry.path());
    }
    //sort images
    std::sort(images.begin(),images.end());
    for (const auto& entry : images) {
        if (entry.extension() != ".jpg" && entry.extension()!=".png")  continue;
        if(filenames.contains(entry. filename().string())) {
            std::cout<<"Skipping already processed "<<entry.filename().string()<<std::endl;
            continue;
        }
        std::cout<<"Processing "<<entry.filename().string()<<std::endl;
        std::map<std::string,maxinit<double>> bestTimes;
        cv::Mat im,imcolorin=cv::imread(entry.string());
        auto sizex=imcolorin.cols*scale;
        auto sizey=imcolorin.rows*scale;
        cv::resize(imcolorin,imcolorin,cv::Size(sizex,sizey));
        cv::cvtColor(imcolorin,im,cv::COLOR_BGR2GRAY);
        std::cerr<<"Size="<<im.cols<<" x "<<im.rows<<std::endl;
        cv::Mat imthres;
        if(!binary)
            cv::adaptiveThreshold(im,imthres,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY_INV,11,5);
        else
            cv::threshold(im,imthres,128,255,cv::THRESH_BINARY);

        //cv::imshow("imx",imthres);cv::waitKey(0);
        for(int i=0;i<testNTimesMain;i++){
            tests(imthres,bestTimes);

            double base;
            if(bestTimes.contains(baseComparison))
                base=bestTimes[baseComparison].val;
            else base=bestTimes.begin()->second.val;
            std::cerr<<"---- Speedup table ----"<<std::endl;
            for(auto&p:bestTimes){
                std::cerr<<"    "<<p.first<<" : "<<p.second.val<<" ms, speedup "<<base/p.second.val<<std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        //save to csv
        if(firstTimeCSV){
            outCSv<<"image,size,";
            for(auto&p:bestTimes){
                outCSv<<p.first<<"_ms,"<<p.first<<"_speedup,";
            }
            outCSv<<std::endl;
            firstTimeCSV=false;
        }

        outCSv<<entry.filename().string()<<","<<im.cols<<"x"<<im.rows<<",";
        double base;
        if(bestTimes.contains(baseComparison))
            base=bestTimes[baseComparison].val;
        else base=bestTimes.begin()->second.val;
        for(auto&p:bestTimes){
            outCSv<<p.second.val<<","<<base/p.second.val<<",";
        }
        outCSv<<std::endl;

    }
    return 0;


}

void tests_findcontour( cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes){
    std::vector<std::vector<cv::Point>> contours;
    Timer TheTimer;
    for(int i=0;i<testNTimes;i++){
        //now find contours
        cv::Mat imthres2=imthres.clone();
        TheTimer.start();
        cv::findContours(imthres2,contours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
        TheTimer.end();

    }
    bestTimes["01::findContours"]=std::min(TheTimer.getBestTime() ,bestTimes["01::findContours"].val);
    size_t nc=0;
    for(auto&c:contours) if(c.size()>=GlobalcontourSizeThres) nc++;
    std::cerr<<"01::findContours  Time: "<<TheTimer.getBestTime()<<" ms. Found "<<nc<<" contours"<<std::endl;
}


void test_truco(cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes){
    std::vector<std::vector<cv::Point>> TRUContours;
    for(int nt=minThreads;nt<=maxThreads ;nt++){
        Timer TheTimer;
        for(int i=0;i<testNTimes;i++){
            TRUContours.clear();
            CvNThreadScope scope(nt);
            TheTimer.start();
            cv::findTRUContours(imthres,TRUContours,GlobalcontourSizeThres);
            TheTimer.end();
        }
        auto stn=std::to_string(nt);
        while(stn.size()<2) stn="0"+stn;//pad with 0
        bestTimes["04::finTRUContours("+stn+")"]=std::min( TheTimer.getBestTime(),bestTimes["04::finTRUContours("+stn+")"].val);
        std::cerr<<"04::finTRUContours("+stn+") Time: "<<TheTimer.getBestTime()<<" ms. Found "<<TRUContours.size()<<" contours"<<std::endl;

        if(TEST_TRUCO_DEV_SAME_AS_SUZUKI) {
            std::vector<std::vector<cv::Point>> contoursSuzuki;
            cv::findContours(imthres.clone(),contoursSuzuki,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
            if(!areEquivalent(contoursSuzuki,TRUContours)){
                std::cout<<"⚠️ Differences found between Suzuki and findUContours with "<<nt<<" threads."<<std::endl;
                //stop the program here
                cv::imshow("ERROR",imthres);cv::waitKey(0);
            }
            else{
                std::cout<<"✅ No differences found between Suzuki and findUContours with "<<nt<<" threads."<<std::endl;
            }
            //now, draw all contours in an outout image in color
            cv::Mat color;
            cv::cvtColor(imthres,color,cv::COLOR_GRAY2BGR);
            for(auto &contour:contoursSuzuki){
                cv::Scalar rcolor(rand()%256,rand()%256,rand()%256);
                for(auto &pt:contour){
                    color.at<cv::Vec3b>(pt)=cv::Vec3b(rcolor[0],rcolor[1],255);//RED
                }
            }
        }
    }
}

void test_findcontourlinks(cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes){
    std::vector<std::vector<cv::Point>> contours;
    Timer TheTimer;
    for(int i=0;i<testNTimes;i++){
        //now find contours
        cv::Mat imthres2=imthres.clone();
        TheTimer.start();
        cv::findContoursLinkRuns(imthres2,contours);
        TheTimer.end();
    }
    bestTimes["02::findContoursLinkRuns"]=std::min(TheTimer.getBestTime(),bestTimes["02::findContoursLinkRuns"].val);
    size_t nc=0;
    for(auto&c:contours) if(c.size()>=GlobalcontourSizeThres) nc++;
    std::cerr<<"02::findContoursLinkRuns  Time: "<<TheTimer.getBestTime()<<" ms. Found "<<nc<<" contours"<<std::endl;
}

void test_garcia(cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes){
    std::cerr<<"Running OpenCV GarciaEtAl..."<<std::endl;
    std::vector<std::vector<cv::Point>> contours;
    Timer TheTimer;
    btomp::BorderTrackerOMP tracker(imthres.cols,imthres.rows);
    for(int i=0;i<testNTimes;i++){
        //now find contours
        cv::Mat imthres2=imthres.clone();
        imthres2/=255;
        TheTimer.start();
        contours=tracker.findContours(imthres2);
        TheTimer.end();
    }
    bestTimes["03::GarciaEtAl"]=std::min(TheTimer.getBestTime(),bestTimes["03::GarciaEtAl"].val);
    size_t nc=0;
    for(auto&c:contours) if(c.size()>=GlobalcontourSizeThres) nc++;
    std::cerr<<"03::GarciaEtAl contour  Time: "<<TheTimer.getBestTime()<<" ms. Found "<<nc<<" contours"<<std::endl;
}

void tests( cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes){
    std::cerr<<"*******************************************"<<std::endl;
    std::vector<std::function<void(cv::Mat&,std::map<std::string,maxinit<double>> &)>> testsToRun;
    // cv::imshow("imthres",imthres);cv::waitKey(0);
    // cv::imwrite("/home/salinas/imthres.png",imthres);
    bool normaltest=true;
    //set opencv num of threads to 1
    //cv::setNumThreads(1);

    //threshold using adaptive threshold.
    //measure time
    ////////////////////////////////////////
    /// OpenCV findContours
    if(TEST_FINDCONTOUR  ){
        testsToRun.push_back( [](cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes)        {
            tests_findcontour(imthres,bestTimes);
        });
    }


    ////////////////////////////////////////
    /// OpenCV  findTRUContours
    if(TEST_TRUCO){
        testsToRun.push_back( [](cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes)        {
            test_truco(imthres,bestTimes);
        });
    }



    ////////////////////////////////////////
    /// OpenCV findContoursLinkRuns
    if(TEST_FINDCONTOURRUNLINKS){
        testsToRun.push_back( [](cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes)        {
            test_findcontourlinks(imthres,bestTimes);
        });
    }

    ////////////////////////////////////////
    /// OpenCV GarciaEtAl
    if(TEST_GARCIA  ){
        testsToRun.push_back( [](cv::Mat &imthres,std::map<std::string,maxinit<double>> &bestTimes)        {
            test_garcia(imthres,bestTimes);
        });

    }
    //randomly sort the testsToRun
    std::shuffle(testsToRun.begin(),testsToRun.end(),std::mt19937(std::random_device()()));

    //test all runs
    for(auto &test:testsToRun){
        test(imthres,bestTimes);
    }
}


bool areEquivalent(const std::vector<std::vector<cv::Point> > &cont1, const std::vector<std::vector<cv::Point> > &cont2){

    auto contoursMatch_strict=[](const std::vector<cv::Point>& cont1,const std::vector<cv::Point>& cont2)->bool{
        if(cont1.size()!=cont2.size())
            return false;

        for (size_t p = 0; p < cont1.size(); ++p) {
            if (cont1[p]!= cont2[p]){
                return false;
            }
        }
        return true;

    };

    auto __uHashContour  =[](const std::vector<cv::Point>& contour)   ->uint64_t{

        auto hash_mix = [](uint64_t x) -> uint64_t {
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            return x;
        };

        const size_t n = contour.size();
        if (n == 0) return hash_mix(0);

        // -- Step 1: find the lexicographically smallest cyclic rotation --------
        // O(n^2) worst case, but effectively O(n) in practice because the minimum
        // contour point is almost always unique. For degenerate cases with many
        // ties (all points equal, etc.) this still terminates correctly.
        auto less_pt = [](const cv::Point& a, const cv::Point& b) {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        };
        size_t start = 0;
        for (size_t cand = 1; cand < n; ++cand) {
            for (size_t k = 0; k < n; ++k) {
                const cv::Point& a = contour[(start + k) % n];
                const cv::Point& b = contour[(cand  + k) % n];
                if (a == b) continue;
                if (less_pt(b, a)) start = cand;
                break;
            }
        }

        // -- Step 2: fold the points starting at `start` through an --------------
        // -- order-SENSITIVE combiner (so [A,B,C] and [A,C,B] differ). ----------
        // The cyclic rotation invariance comes only from the canonical start
        // index chosen above; the combiner itself is fully order-sensitive.
        uint64_t h = hash_mix(static_cast<uint64_t>(n));
        for (size_t i = 0; i < n; ++i) {
            const cv::Point& pt = contour[(start + i) % n];
            // Pack (x,y) losslessly into 64 bits; cast through uint32_t so that
            // any negative coordinate doesn't sign-extend and clobber the x half.
            uint64_t packed = (static_cast<uint64_t>(static_cast<uint32_t>(pt.x)) << 32)
                              |  static_cast<uint64_t>(static_cast<uint32_t>(pt.y));
            h = hash_mix(h ^ hash_mix(packed));
        }
        return h;
    };

    struct cinfo
    {
        int ntimes=0;
        std::vector<cv::Point> contour;
    };
    std::map<uint64,cinfo> hashes1,hashes2;;
    for(auto &contour:cont1){
        auto hash=__uHashContour(contour);
        hashes1[hash].contour= contour;
        hashes1[hash].ntimes++;

    }
    for(auto &contour:cont2){
        auto hash=__uHashContour(contour);
        hashes2[hash].contour= contour;
        hashes2[hash].ntimes++;
    }

    //compare the maps, they should contain the same keys with the same number of times
    if(hashes1.size()!=hashes2.size()){
//        std::cout<<"Different number of unique contours: "<<hashes1.size()<<" vs "<<hashes2.size()<<std::endl;
        return false;
    }
    for(auto it1=hashes1.begin(),it2=hashes2.begin();it1!=hashes1.end() && it2!=hashes2.end();it1++,it2++){
        if(it1->first!=it2->first){
//            std::cout<<"Different contour hash: "<<it1->first<<" vs "<<it2->first<<std::endl;
            return false;
        }
        if(it1->second.ntimes!=it2->second.ntimes){
  //          std::cout<<"Different number of times for contour hash "<<it1->first<<": "<<it1->second.ntimes<<" vs "<<it2->second.ntimes<<std::endl;
            return false;
        }
    }
    return true;
}

std::optional< std::pair<int,cv::Mat> > contourDiffImage(cv::Mat &inThres,  std::vector<std::vector<cv::Point>> &cont1,std::vector<std::vector<cv::Point>> &cont2){

    cv::Mat thres_io_3channels;
    if(inThres.channels()==1)
        cv::cvtColor(inThres,thres_io_3channels,cv::COLOR_GRAY2BGR);
    else
        thres_io_3channels=inThres.clone();

    int ndiff=0;
    auto __uHashContour=[](const std::vector<cv::Point>& contour) ->uint64_t{
        auto hash_mix = [](uint64_t x) -> uint64_t {
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            return x;
        };

        uint64_t combinedPointHash = 0;
        for (const auto& pt : contour) {
            // Map (x, y) to a unique 64-bit seed
            uint64_t seed = (static_cast<uint64_t>(pt.x) * 0x1f1f1f1f1f1f1f1fULL) ^ static_cast<uint64_t>(pt.y);
            combinedPointHash += hash_mix(seed);
        }

        // Mix the contour size with the accumulated point hash
        // We mix the size first so it acts as a unique 'salt' for the final transform
        uint64_t finalHash = hash_mix(combinedPointHash ^ hash_mix(contour.size()));

        return finalHash;
    };


    std::map<uint64,std::vector<cv::Point>> hashes1,hashes2;;
    for(auto &contour:cont1)
        hashes1[__uHashContour(contour)]=contour;
    for(auto &contour:cont2)
        hashes2[__uHashContour(contour)]=contour;


    //find the contours that are not in both
    for(auto &pair:hashes1){//element in cont and not in cont2
        if(hashes2.find(pair.first)==hashes2.end()){
            ndiff++;
            //create a random color
            cv::Scalar rcolor(rand()%256,rand()%256,rand()%256);
            //print the starting point of the ccontour
            //    std::cout<<"A "<<pair.second.size()<<" start="<<pair.second[0]<<std::endl;

            //draw in red
            for(auto &pt:pair.second){
                thres_io_3channels.at<cv::Vec3b>(pt)=cv::Vec3b(rcolor[0],rcolor[1],255);//RED
            }
        }
    }
    for(auto &pair:hashes2){//element in cont2 and not in cont
        if(hashes1.find(pair.first)==hashes1.end()){
            ndiff++;
            //draw in red
            //print the starting point of the ccontour
            //    std::cout<<"B "<<pair.second.size()<<" start="<<pair.second[0]<<std::endl;
            for(auto &pt:pair.second){
                thres_io_3channels.at<cv::Vec3b>(pt)=cv::Vec3b(255,0,0);//BLUE
            }
        }
    }
    if(ndiff==0)return {};
    else return std::pair<int,cv::Mat>{ndiff,thres_io_3channels};
}

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

//キャリブレーション関係変数
#define IMAGE_NUM  (10)         /* 画像数 */
#define PAT_ROW    (7)          /* パターンの行数 */
#define PAT_COL    (10)         /* パターンの列数 */
#define PAT_SIZE   (PAT_ROW*PAT_COL)
#define ALL_POINTS (IMAGE_NUM*PAT_SIZE)
#define CHESS_SIZE (24.0)       /* パターン1マスの1辺サイズ[mm] */

#define PROJECTOR_WIDTH (1280) //プロジェクタ解像度
#define PROJECTOR_HEIGHT (800) //プロジェクタ解像度
//1280 * 800の場合
#define PROJECTOR_CHESS_SIZE_WIDTH (103) /* プロジェクタ画像上でのパターン1マスの幅サイズ[pixel] */
#define PROJECTOR_CHESS_SIZE_HEIGHT (90) /* プロジェクタ画像上でのパターン1マスの高さサイズ[pixel] */ //目測値

//入力画像配列
vector<Mat> checkerimg_src;

//---二値化画像生成用変数---//
Mat bin_b, bin_r; //R,Bプレーンの結果

//---カメラキャリブレーション用変数---//
//チェッカパターンの交点の数
CvSize pattern_size = cvSize (PAT_COL, PAT_ROW);
//交点の3次元座標
CvPoint3D32f objects[ALL_POINTS];
//交点のカメラ画像上の座標
CvPoint2D32f *corners = (CvPoint2D32f *) cvAlloc (sizeof (CvPoint2D32f) * ALL_POINTS);
//Calibration関数用配列
CvMat object_points;//(共通)
CvMat image_points;
CvMat point_counts;//(共通)

//---プロジェクタキャリブレーション用変数---//
//プロジェクタで投影しているチェッカの交点のカメラ画像上の座標
CvPoint2D32f *proj_corners = (CvPoint2D32f *) cvAlloc (sizeof (CvPoint2D32f) * ALL_POINTS);
int p_count[IMAGE_NUM];

//Calibration関数用配列
CvMat proj_image_points;


//---各種名前---//
const char* windowname_proj = "Projector Pattern";
const char* windowname_cam = "Camera Pattern";

const string	windowNameUnd_R = "Undistorted RPlane Image";
const string	windowNameUnd_B = "Undistorted BPlane Image";
const string	windowNameUnd_src = "Undistorted src Image";

const string srcfoldername = "../src_image_valid/";
const string rPlanefoldername = "../Rplane_image/";
const string bPlanefoldername = "../Bplane_image/";
const string undistortedfoldername = "../undistorted_image/";


//カメラ用チェッカパターン画像配列(Rplane画像)
vector<Mat> checkerimg_cam;
//カメラ用チェッカパターン画像配列(カメラの歪み補正後)
vector<Mat> checkerimg_cam_undist;

//プロジェクタ用チェッカパターン画像配列(Bplane画像)
vector<Mat> checkerimg_proj;
//プロジェクタ用チェッカパターン画像配列(カメラの歪み補正後)
vector<Mat> checkerimg_proj_undistCamera;

////プロジェクタ用チェッカパターン画像配列(カメラとプロジェクタの歪み補正後)
//vector<Mat> checkerimg_proj_undistCameraProj;
//
////プロジェクタキャリブレーション用 カメラ画像座標でのチェッカ交点座標
//vector<vector<Point2f>> cameraImagePoints_proj(IMAGE_NUM);
//プロジェクタキャリブレーション用 プロジェクタ画像上でのチェッカボードの交点座標
vector<vector<Point2f>> projImagePoints_checkerboard;
//
//
///*プロジェクタパラメータ行列*/
//cv::Mat				projMatrix;		// 内部パラメータ行列
//cv::Mat				projDistCoeffs;		// レンズ歪み行列
//cv::vector<cv::Mat>	projRotationVectors;	// 撮影画像ごとに得られる回転ベクトル
//cv::vector<cv::Mat>	projTranslationVectors;	// 撮影画像ごとに得られる平行移動ベクトル

/*カメラパラメータ*/
CvMat *camera_intrinsic = cvCreateMat (3, 3, CV_32FC1);
CvMat *camera_rotation = cvCreateMat (1, 3, CV_32FC1);
CvMat *camera_translation = cvCreateMat (1, 3, CV_32FC1);
CvMat *camera_distortion = cvCreateMat (1, 4, CV_32FC1);
/*プロジェクタパラメータ*/
CvMat *proj_intrinsic = cvCreateMat (3, 3, CV_32FC1);
CvMat *proj_rotation = cvCreateMat (1, 3, CV_32FC1);
CvMat *proj_translation = cvCreateMat (1, 3, CV_32FC1);
CvMat *proj_distortion = cvCreateMat (1, 4, CV_32FC1);




//フォルダ名と画像番号からファイル名を生成
string get_capImgFileName(string foldername, int num){
	string str1 = "cap";
	string str2 = to_string(num);
	string str3 = ".png";
	string filename = foldername + str1 + str2 + str3;

	return filename;
}

Mat getRedImage(Mat srcimg){
	Mat result = srcimg.clone();

	for(int y = 0; y < result.rows; y++){
		for(int x = 0; x < result.cols; x++){
			int index = y * result.step + x * result.channels();
			result.data[index + 0] = 0;
			result.data[index + 1] = 0;
		}
	}

	return result;
}

Mat getBlueImage(Mat srcimg){
	Mat result = srcimg.clone();

	for(int y = 0; y < result.rows; y++){
		for(int x = 0; x < result.cols; x++){
			int index = y * result.step + x * result.channels();
			result.data[index + 1] = 0;
			result.data[index + 2] = 0;
		}
	}

	return result;
}

void threshold_div(Mat srcimg, int dev, int flag){
	Mat src = srcimg.clone();

	//IplImageへ変換
	IplImage ipl = src;

	int roi_w = (int)(src.cols / dev);
	int roi_h = (int)(src.rows / dev);

	for(int y = 0; y < src.rows; y+=roi_h){
		for(int x = 0; x < src.cols; x+= roi_w){
			//ROIをセット
			cvSetImageROI(&ipl, Rect(x, y, roi_w,roi_h));

			//二値化
			cvThreshold(&ipl, &ipl, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
			//ROI解除
			cvResetImageROI(&ipl);
		}
	}

	//Matへ変換
	Mat result = srcimg.clone();
    result = cv::cvarrToMat(&ipl);  // データをコピーする
	if (flag) bin_r = result.clone(); //1:b 0:r
	else bin_b = result.clone();
  //CV_Assert(reinterpret_cast<uchar*>(ipl.imageData) != result.data);

	waitKey(0);

}

void RPlanefilter(){
	Mat dst_r, gray_r;
	for(int i = 0; i < IMAGE_NUM; i++){
		//色分離
		Mat r_image = getRedImage(checkerimg_src[i]);
		//平滑化(バイラテラルフィルタ)
		cv::bilateralFilter(r_image, dst_r, 11, 40, 200);
		//グレースケール
		cvtColor(dst_r, gray_r, CV_BGR2GRAY);
		//二値化
		threshold_div(gray_r, 6, 1);
		//cv::threshold(gray_r, bin_r, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
		//保存
		imwrite(get_capImgFileName(rPlanefoldername, i), bin_r);
	}
}

void BPlanefilter(){
	Mat dst_b,  gray_b;
	for(int i = 0; i < IMAGE_NUM; i++){
		//色分離
		Mat r_image = getBlueImage(checkerimg_src[i]);
		//平滑化(バイラテラルフィルタ)
		cv::bilateralFilter(r_image, dst_b, 11, 40, 200);
		//グレースケール
		cvtColor(dst_b, gray_b, CV_BGR2GRAY);
		//二値化
		//bのみ6分割にして二値化
		threshold_div(gray_b, 6, 0);
		//cv::threshold(gray_b, bin_b, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

		//反転
		//Mat rev = ~bin_b;
		//保存
		//imwrite(get_capImgFileName(bPlanefoldername, i), rev);
		imwrite(get_capImgFileName(bPlanefoldername, i), bin_b);
	}
}

//void findProjectorChessboardCorners(){
//	//表示用ウィンドウ
//	cv::namedWindow( windowname_proj, CV_WINDOW_AUTOSIZE );
//
//	// チェックパターンの交点座標を求め，imagePointsに格納する
//	for(int i = 0; i < IMAGE_NUM; i++){
//		cout << "Find corners from image " << i;
//		bool result = findChessboardCorners(checkerimg_proj_undistCamera[i], patternSize, cameraImagePoints_proj[i]);
//		//bool result = findChessboardCorners(checkerimg_proj[i], patternSize, cameraImagePoints_proj[i]);
//		if(result){
//			cout << " ... All corners found." << endl;
//			cout << cameraImagePoints_proj[i].size() << endl;
//			//cv::Mat grayImg;
//			//cv::cvtColor(checkerimg_proj_undistCamera[i], grayImg, CV_BGR2GRAY);	//グレースケール化
//			cv::cornerSubPix(checkerimg_proj_undistCamera[i], cameraImagePoints_proj[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //コーナー位置をサブピクセル精度に修正
//			//cv::cornerSubPix(checkerimg_proj[i], cameraImagePoints_proj[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //コーナー位置をサブピクセル精度に修正
//			// 検出点を描画する
//			//cv::drawChessboardCorners( checkerimg_proj_undistCamera[i], patternSize, ( cv::Mat )( cameraImagePoints_proj[i] ), true );
//
//			//表示
//			cv::imshow( windowname_proj, checkerimg_proj_undistCamera[i] );
//			//cv::imshow( windowname_proj, checkerimg_proj[i] );
//			//保存
//			imwrite(get_capImgFileName(bPlanefoldername, i), checkerimg_proj_undistCamera[i]);
//			//imwrite(get_capImgFileName(bPlanefoldername, i), checkerimg_proj[i]);
//			waitKey( 0 );
//		}else{
//			cout << " ... at least 1 corner not found." << endl;
//			cout << cameraImagePoints_proj[i].size() << endl;
//			waitKey( 1 );
//		}
//	}
//}

void findProjectorChessboardCorners2()
{
	int i, j;
	int corner_count, found;
	//int p_count[IMAGE_NUM];
	IplImage *src_img[IMAGE_NUM];

	// (1)キャリブレーション画像の読み込み
	for (i = 0; i < IMAGE_NUM; i++) {
		std::string filename = get_capImgFileName(bPlanefoldername, i);
		if ((src_img[i] = cvLoadImage (filename.data(), CV_LOAD_IMAGE_COLOR)) == NULL) {
			fprintf (stderr, "cannot load image file : %s\n", filename);
		}
		cvNot(src_img[i], src_img[i]);
	}
	// (3)チェスボード（キャリブレーションパターン）のコーナー検出
	int found_num = 0;
	cvNamedWindow ("Calibration", CV_WINDOW_AUTOSIZE);
	for (i = 0; i < IMAGE_NUM; i++) {
	found = cvFindChessboardCorners (src_img[i], pattern_size, &proj_corners[i * PAT_SIZE], &corner_count);
	fprintf (stderr, "%02d...", i);
	if (found) {
		fprintf (stderr, "ok\n");
		found_num++;
	}
	else {
		fprintf (stderr, "fail\n");
	}
	// (4)コーナー位置をサブピクセル精度に修正，描画
	IplImage *src_gray = cvCreateImage (cvGetSize (src_img[i]), IPL_DEPTH_8U, 1);
	cvCvtColor (src_img[i], src_gray, CV_BGR2GRAY);
	cvFindCornerSubPix (src_gray, &proj_corners[i * PAT_SIZE], corner_count,
						cvSize (3, 3), cvSize (-1, -1), cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	//cvDrawChessboardCorners (src_img[i], pattern_size, &proj_corners[i * PAT_SIZE], corner_count, found);
	//p_count[i] = corner_count;
	cvShowImage ("Calibration", src_img[i]);
	cvWaitKey (0);
	}
	cvDestroyWindow ("Calibration");

	if (found_num != IMAGE_NUM)
		cout << "fail: found_num != IMAGE_NUM" << endl;
}

void CameraCalibration(){
	/*カメラパラメータ行列*/
	cv::Mat				cameraMatrix;		// 内部パラメータ行列
	cv::Mat				cameraDistCoeffs;		// レンズ歪み行列
	cv::vector<cv::Mat>	cameraRotationVectors;	// 撮影画像ごとに得られる回転ベクトル
	cv::vector<cv::Mat>	cameraTranslationVectors;	// 撮影画像ごとに得られる平行移動ベクトル

	//チェッカパターンの交点の数
	const Size patternSize(10, 7); 
	//カメラ用チェッカパターンの左上を原点とする、チェッカ交点の世界座標
	vector<vector<Point3f>> worldPoints(IMAGE_NUM);
	//カメラキャリブレーション用 カメラ画像座標でのチェッカ交点座標
	vector<vector<Point2f>> cameraImagePoints_camera(IMAGE_NUM);

	// 対応するワールド座標系パターン
	TermCriteria criteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.001 );  


	//表示用ウィンドウ
	cv::namedWindow( windowname_cam, CV_WINDOW_AUTOSIZE );

	// チェックパターンの交点座標を求め，imagePointsに格納する
	for(int i = 0; i < IMAGE_NUM; i++){
		cout << "Find corners from image " << i;
		bool result = findChessboardCorners(checkerimg_cam[i], patternSize, cameraImagePoints_camera[i]);
		if(result){
			cout << " ... All corners found." << endl;
			cout << cameraImagePoints_camera[i].size() << endl;
			//cv::Mat grayImg;
			//cv::cvtColor(checkerimg_cam[i], grayImg, CV_BGR2GRAY);	//グレースケール化
			cv::cornerSubPix(checkerimg_cam[i], cameraImagePoints_camera[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //コーナー位置をサブピクセル精度に修正
			// 検出点を描画する
			//cv::drawChessboardCorners( checkerimg_cam[i], patternSize, ( cv::Mat )( cameraImagePoints_camera[i] ), true );
			//表示
			cv::imshow( windowname_cam, checkerimg_cam[i] );
			waitKey( 0 );
		}else{
			cout << " ... at least 1 corner not found." << endl;
			cout << cameraImagePoints_camera[i].size() << endl;
			waitKey( 1 );
		}
	}

//カメラ用チェッカパターンの左上を原点とする、チェッカ交点の世界座標を決める
	for( int i = 0; i < IMAGE_NUM; i++ ) {
		for( int j = 0 ; j < patternSize.area(); j++ ) {
			worldPoints[i].push_back( cv::Point3f(	static_cast<float>( j % patternSize.width *  CHESS_SIZE), 
				static_cast<float>( j / patternSize.width * CHESS_SIZE ), 
						  0.0 ) );
		}
	}

	// これまでの値を使ってキャリブレーション
	cv::calibrateCamera( worldPoints, cameraImagePoints_camera, checkerimg_cam[0].size(), cameraMatrix, cameraDistCoeffs, 
			     cameraRotationVectors, cameraTranslationVectors );
	cout << "Camera parameters have been estimated" << endl << endl;

	/*出力*/
	cout << "<<CameraCalibration result>>" << endl;
	cout << "*********Camera Parameters*********" << endl;
	cout << "Camera Matrix:\n" << cameraMatrix << endl;
	cout << "Camera DistCoeffs:\n" << cameraDistCoeffs << endl;
	cout << "index 0 --- Camera Rotate:\n" << cameraRotationVectors.at(0) << endl;
	cout << "index 0 --- Camera Translation:\n" << cameraTranslationVectors.at(0) << endl;

	//Kinectの画像は歪み補正済み？？
	//Rプレーン画像とBPlane画像からカメラの歪みを補正し、補正後の画像を保存
  	cout << "Undistorted RPlane and BPlane images" << endl;
	cv::Mat	undistorted_r, undistorted_b, undistorted_src;
	cv::namedWindow( windowNameUnd_R );
	cv::namedWindow( windowNameUnd_B );
	cv::namedWindow( windowNameUnd_src );
	for( int i = 0; i < IMAGE_NUM; i++ ) {
		cv::undistort( checkerimg_cam[i], undistorted_r, cameraMatrix, cameraDistCoeffs );
		cv::undistort( checkerimg_proj[i], undistorted_b, cameraMatrix, cameraDistCoeffs );
		cv::undistort( checkerimg_src[i], undistorted_src, cameraMatrix, cameraDistCoeffs );

		cv::imshow( windowNameUnd_R, undistorted_r );
		cv::imshow( windowNameUnd_B, undistorted_b );
		cv::imshow( windowNameUnd_src, undistorted_src );

		cv::imshow( windowname_cam, checkerimg_cam[i] );
		cv::imshow( windowname_proj, checkerimg_proj[i] );

		//checkerimg_cam_undist.push_back(undistorted_r);
		//checkerimg_proj_undistCamera.push_back(undistorted_b);
		//保存
		//imwrite(get_capImgFileName(rPlanefoldername, i), undistorted_r);
		//imwrite(get_capImgFileName(bPlanefoldername, i), undistorted_b);
		imwrite(get_capImgFileName(undistortedfoldername, i), undistorted_src);


		waitKey( 0 );
	}
}

//CV2系を使ったもの
int CameraCalibration2(){
  int i, j, k;
  int corner_count, found;
  IplImage *src_img[IMAGE_NUM];

  // (1)キャリブレーション画像の読み込み
  for (i = 0; i < IMAGE_NUM; i++) {
	  std::string filename = get_capImgFileName(rPlanefoldername, i);
	  if ((src_img[i] = cvLoadImage (filename.data(), CV_LOAD_IMAGE_COLOR)) == NULL) {
		  fprintf (stderr, "cannot load image file : %s\n", filename);
    }
  }

  // (2)3次元空間座標の設定
  for (i = 0; i < IMAGE_NUM; i++) {
    for (j = 0; j < PAT_ROW; j++) {
      for (k = 0; k < PAT_COL; k++) {
        objects[i * PAT_SIZE + j * PAT_COL + k].x = j * CHESS_SIZE;
        objects[i * PAT_SIZE + j * PAT_COL + k].y = k * CHESS_SIZE;
        objects[i * PAT_SIZE + j * PAT_COL + k].z = 0.0;
      }
    }
  }
  cvInitMatHeader (&object_points, ALL_POINTS, 3, CV_32FC1, objects);

  // (3)チェスボード（キャリブレーションパターン）のコーナー検出
  int found_num = 0;
  cvNamedWindow ("Calibration", CV_WINDOW_AUTOSIZE);
  for (i = 0; i < IMAGE_NUM; i++) {
    found = cvFindChessboardCorners (src_img[i], pattern_size, &corners[i * PAT_SIZE], &corner_count);
    fprintf (stderr, "%02d...", i);
    if (found) {
      fprintf (stderr, "ok\n");
      found_num++;
    }
    else {
      fprintf (stderr, "fail\n");
    }
    // (4)コーナー位置をサブピクセル精度に修正，描画
    IplImage *src_gray = cvCreateImage (cvGetSize (src_img[i]), IPL_DEPTH_8U, 1);
    cvCvtColor (src_img[i], src_gray, CV_BGR2GRAY);
    cvFindCornerSubPix (src_gray, &corners[i * PAT_SIZE], corner_count,
                        cvSize (3, 3), cvSize (-1, -1), cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
    cvDrawChessboardCorners (src_img[i], pattern_size, &corners[i * PAT_SIZE], corner_count, found);
    p_count[i] = corner_count;
    cvShowImage ("Calibration", src_img[i]);
    cvWaitKey (0);
  }
  cvDestroyWindow ("Calibration");

  if (found_num != IMAGE_NUM)
    return -1;
  cvInitMatHeader (&image_points, ALL_POINTS, 1, CV_32FC2, corners);
  cvInitMatHeader (&point_counts, IMAGE_NUM, 1, CV_32SC1, p_count);

  // (5)内部パラメータ，歪み係数の推定
  cvCalibrateCamera2 (&object_points, &image_points, &point_counts, cvSize (640, 480), camera_intrinsic, camera_distortion);

  // (6)外部パラメータの推定
  CvMat sub_image_points, sub_object_points;
  int base = 0;
  cvGetRows (&image_points, &sub_image_points, base * PAT_SIZE, (base + 1) * PAT_SIZE);
  cvGetRows (&object_points, &sub_object_points, base * PAT_SIZE, (base + 1) * PAT_SIZE);
  cvFindExtrinsicCameraParams2 (&sub_object_points, &sub_image_points, camera_intrinsic, camera_distortion, camera_rotation, camera_translation);

  // (7)XMLファイルへの書き出し
  CvFileStorage *fs;
  fs = cvOpenFileStorage ("camera2.xml", 0, CV_STORAGE_WRITE);
  cvWrite (fs, "intrinsic", camera_intrinsic);
  cvWrite (fs, "rotation", camera_rotation);
  cvWrite (fs, "translation", camera_translation);
  cvWrite (fs, "distortion", camera_distortion);
  cvReleaseFileStorage (&fs);

	//出力
    cout << "<<---CameraCalibration2 result--->>" << endl;
	cout << "*********Camera Parameters*********" << endl;
	cout << "Camera Matrix:\n" << cv::Mat(camera_intrinsic) << endl;
	cout << "Camera DistCoeffs:\n" << cv::Mat(camera_distortion) << endl;
	cout << "index 0 --- Camera Rotate:\n" << cv::Mat(camera_rotation) << endl;
	cout << "index 0 --- Camera Translation:\n" << cv::Mat(camera_translation) << endl;

   //歪み補正
    std::cout << "Undistorted images" << std::endl;
	const char* windowNameUnd = "Undistoted Image";
	const char* windowNameSrc = "Src Image";

	cv::namedWindow( windowNameUnd );
	for( int i = 0; i < IMAGE_NUM; i++ ) {
		IplImage src = checkerimg_src[i];
		IplImage Rplane = checkerimg_cam[i];
		IplImage Bplane = checkerimg_proj[i];
		IplImage*	undistorted_src = cvCreateImage(cvSize(src.width,src.height), src.depth, src.nChannels);
		IplImage*	undistorted_r = cvCreateImage(cvSize(Rplane.width, Rplane.height), Rplane.depth, Rplane.nChannels);
		IplImage*	undistorted_b = cvCreateImage(cvSize(Bplane.width,Bplane.height), Bplane.depth, Bplane.nChannels);

		cvUndistort2(&src, undistorted_src, camera_intrinsic, camera_distortion); //元画像
		cvUndistort2(&Rplane, undistorted_r, camera_intrinsic, camera_distortion); //Rplane画像
		cvUndistort2(&Bplane, undistorted_b, camera_intrinsic, camera_distortion); //Bplane画像

		//表示
		cvShowImage( windowNameUnd, undistorted_src );
		cvShowImage( windowNameSrc, src_img[i]);
		//保存
		cvSaveImage(get_capImgFileName(undistortedfoldername, i).data(), undistorted_src);
		cvSaveImage(get_capImgFileName(rPlanefoldername, i).data(), undistorted_r);
		cvSaveImage(get_capImgFileName(bPlanefoldername, i).data(), undistorted_b);

		cv::waitKey( 0 );
	}


  for (i = 0; i < IMAGE_NUM; i++) {
    cvReleaseImage (&src_img[i]);
  }
  return 0;
}

Point2f multHomography(Mat h, CvPoint2D32f p)
{
	float x = (float)(h.at<double>(0) * p.x + h.at<double>(1) * p.y + h.at<double>(2));
	float y = (float)(h.at<double>(3) * p.x + h.at<double>(4) * p.y + h.at<double>(5));
	float z = (float)( h.at<double>(6) * p.x + h.at<double>(7) * p.y + h.at<double>(8));

	x = x/z;
	y = y/z;

	return Point2f(x, y);
}

int main( int argc, char** argv )
{
	//(1) 画像のロード:src
	for(int i = 0; i < IMAGE_NUM; i++){
		checkerimg_src.push_back(imread(get_capImgFileName(srcfoldername, i)));
	}

	//(2) R,Bプレーンで分離→二値化→保存
	RPlanefilter();
	BPlanefilter();

	//分離した画像のロード:rplane bplane
	for(int i = 0; i < IMAGE_NUM; i++){
		//グレースケールで読み込む imread-> flag: 正:3channel 0:grayscale 負:そのまま
		checkerimg_cam.push_back(imread(get_capImgFileName(rPlanefoldername, i), 0));
		checkerimg_proj.push_back(imread(get_capImgFileName(bPlanefoldername, i), 0));
	}

	//(3) カメラのパラメータ推定及びカメラの歪み補正
	//CameraCalibration();
	CameraCalibration2();

	//(4) カメラ・プロジェクタ間のホモグラフィ行列の推定(チェッカパターンの四隅から)
	//四隅のインデックス
	int index0 = 0;
	int index1 = pattern_size.width - 1;
	int index2 = pattern_size.width * (pattern_size.height - 1);
	int index3 = PAT_SIZE -1;

	//画像(カメラの歪み補正後)のロード:undistorted_r undistorted_b
	for(int i = 0; i < IMAGE_NUM; i++){
		checkerimg_cam_undist.push_back(imread(get_capImgFileName(rPlanefoldername, i), 0));
		checkerimg_proj_undistCamera.push_back(imread(get_capImgFileName(bPlanefoldername, i), 0));
	}

	//今度はBプレーン画像からコーナー検出
	//findProjectorChessboardCorners();

	findProjectorChessboardCorners2();

	//各画像の四隅を使って初期ホモグラフィ行列Hを推定
	//↓
	//(5) カメラで撮影したチェッカボードの交点をホモグラフィ変換し、プロジェクタ画像座標に変換

	//プロジェクタ画像上の点(dst)
	Point2f corner0_projImg(172, 129);
	Point2f corner1_projImg(1102, 129);
	Point2f corner2_projImg(172, 677);
	Point2f corner3_projImg(1102, 677);

	for(int i = 0; i < IMAGE_NUM; i++)
	{

	//カメラ画像上で検出された点(src)
	Point2f corner0_detect(proj_corners[i * PAT_SIZE + index0].x, proj_corners[i * PAT_SIZE + index0].y);
	Point2f corner1_detect(proj_corners[i * PAT_SIZE + index1].x, proj_corners[i * PAT_SIZE + index1].y);
	Point2f corner2_detect(proj_corners[i * PAT_SIZE + index2].x, proj_corners[i * PAT_SIZE + index2].y);
	Point2f corner3_detect(proj_corners[i * PAT_SIZE + index3].x, proj_corners[i * PAT_SIZE + index3].y);
	//Point2f corner0_detect = cameraImagePoints_proj[i][index0];
	//Point2f corner1_detect = cameraImagePoints_proj[i][index1];
	//Point2f corner2_detect = cameraImagePoints_proj[i][index2];
	//Point2f corner3_detect = cameraImagePoints_proj[i][index3];

		//カメラ画像→プロジェクタ画像：H　とする
		Point2f src_pt[] = {corner0_detect, corner1_detect, corner2_detect, corner3_detect};
		Point2f dst_pt[] = {corner0_projImg, corner1_projImg, corner2_projImg, corner3_projImg};

		//カメラ画像→プロジェクタ画像へのホモグラフィ行列
		Mat H = getPerspectiveTransform(src_pt,dst_pt);

		cout << "\nH_" << i << ": \n" << H << endl;

		//カメラで撮影したチェッカボードの交点をホモグラフィ変換し、プロジェクタ画像座標に変換
		vector<Point2f> checkerBoard_coners;
		for(int j = 0; j <  PAT_SIZE; j++)
		{
			//Point2f dst = multHomography(H, cameraImagePoints_camera[i][j]);
			Point2f dst = multHomography(H,corners[i * PAT_SIZE + j]);
			checkerBoard_coners.push_back(dst);

			//cout << "(" << corners[i * PAT_SIZE + j].x << ", " << corners[i * PAT_SIZE + j].y << ") ---> "
			//	<< "(" << dst.x << ", " << dst.y << ")" << endl;
		}
		 projImagePoints_checkerboard.push_back(checkerBoard_coners);
		 checkerBoard_coners.clear();

	}

	//(6) プロジェクタキャリブレーションの実行 objects --- projImagePoints_checkerboard
	//projImagePoints_checkerboardを変換	
	CvPoint2D32f *dst_corners = (CvPoint2D32f *) cvAlloc (sizeof (CvPoint2D32f) * ALL_POINTS);//交点のプロジェクタ画像上の座標
	for(int i = 0; i < IMAGE_NUM; i++)
	{	
		//デバッグ用
		Mat ProjectionImage(PROJECTOR_HEIGHT, PROJECTOR_WIDTH, CV_8UC3);
		for(int j = 0; j < PAT_SIZE; j++)
		{
			dst_corners[i * PAT_SIZE + j].x = projImagePoints_checkerboard[i][j].x;
			dst_corners[i * PAT_SIZE + j].y = projImagePoints_checkerboard[i][j].y;

			circle(ProjectionImage, projImagePoints_checkerboard[i][j], 3.0, Scalar(0, 0, 255));
		}
		imshow("reprojection", ProjectionImage);
		waitKey(0);
	}
	cvInitMatHeader (&proj_image_points, ALL_POINTS, 1, CV_32FC2, dst_corners);
    cvInitMatHeader (&point_counts, IMAGE_NUM, 1, CV_32SC1, p_count);
	// (5)内部パラメータ，歪み係数の推定
	cvCalibrateCamera2 (&object_points, &proj_image_points, &point_counts, cvSize (PROJECTOR_WIDTH, PROJECTOR_HEIGHT), proj_intrinsic, proj_distortion);

	  // (6)外部パラメータの推定
	  CvMat sub_image_points, sub_object_points;
	  int base = 0;
	  cvGetRows (&proj_image_points, &sub_image_points, base * PAT_SIZE, (base + 1) * PAT_SIZE);
	  cvGetRows (&object_points, &sub_object_points, base * PAT_SIZE, (base + 1) * PAT_SIZE);
	  cvFindExtrinsicCameraParams2 (&sub_object_points, &sub_image_points, proj_intrinsic, proj_distortion, proj_rotation, proj_translation);

	  // (7)XMLファイルへの書き出し
	  CvFileStorage *fs;
	  fs = cvOpenFileStorage ("proj.xml", 0, CV_STORAGE_WRITE);
	  cvWrite (fs, "intrinsic", proj_intrinsic);
	  cvWrite (fs, "rotation", proj_rotation);
	  cvWrite (fs, "translation", proj_translation);
	  cvWrite (fs, "distortion", proj_distortion);
	  cvReleaseFileStorage (&fs);

		//出力
		cout << "<<---ProjectorCalibration2 result--->>" << endl;
		cout << "*********Projector Parameters*********" << endl;
		cout << "Camera Matrix:\n" << cv::Mat(proj_intrinsic) << endl;
		cout << "Camera DistCoeffs:\n" << cv::Mat(proj_distortion) << endl;
		cout << "index 0 --- Camera Rotate:\n" << cv::Mat(proj_rotation) << endl;
		cout << "index 0 --- Camera Translation:\n" << cv::Mat(proj_translation) << endl;

		////歪み補正
		//std::cout << "Undistorted images" << std::endl;
		//const char* windowNameUndR = "Undistoted Image R";
		//const char* windowNameUndB = "Undistoted Image B";
		//const char* windowNameSrcR = "Src Image R";
		//const char* windowNameSrcB = "Src Image B";
		//cv::namedWindow( windowNameUndR );
		//cv::namedWindow( windowNameUndB );
		//cv::namedWindow( windowNameSrcR );
		//cv::namedWindow( windowNameSrcB );

		//for( int i = 0; i < IMAGE_NUM; i++ ) {
		//	IplImage Rplane = checkerimg_cam_undist[i];
		//	IplImage Bplane = checkerimg_proj_undistCamera[i];
		//	IplImage*	undistorted_r = cvCreateImage(cvSize(Rplane.width, Rplane.height), Rplane.depth, Rplane.nChannels);
		//	IplImage*	undistorted_b = cvCreateImage(cvSize(Bplane.width,Bplane.height), Bplane.depth, Bplane.nChannels);

		//	cvUndistort2(&Rplane, undistorted_r, proj_intrinsic, proj_distortion); //Rplane画像
		//	cvUndistort2(&Bplane, undistorted_b, proj_intrinsic, proj_distortion); //Bplane画像

		//	//表示
		//	cvShowImage( windowNameUndR, undistorted_r );
		//	cvShowImage( windowNameUndB, undistorted_b );
		//	cvShowImage( windowNameSrcR, &Rplane );
		//	cvShowImage( windowNameSrcB, &Bplane );
		//	//保存
		//	//cvSaveImage(get_capImgFileName(undistortedfoldername, i).data(), undistorted_src);
		//	//cvSaveImage(get_capImgFileName(rPlanefoldername, i).data(), undistorted_r);
		//	//cvSaveImage(get_capImgFileName(bPlanefoldername, i).data(), undistorted_b);

		//	cv::waitKey( 0 );
		//}


	//(6) プロジェクタキャリブレーションの実行 worldPoints --- projImagePoints_checkerboard
	//cv::calibrateCamera( worldPoints, projImagePoints_checkerboard, Size(PROJECTOR_WIDTH, PROJECTOR_HEIGHT), projMatrix, projDistCoeffs, 
	//		     projRotationVectors, projTranslationVectors );

	////出力
	///*プロジェクタ*/
	//cout << "*********Projector Parameters*********" << endl;
	//cout << "Projector Matrix:\n" << projMatrix << endl;
	//cout << "Projector DistCoeffs:\n" << projDistCoeffs << endl;
	//cout << "index 0 --- Projector Rotate:\n" << projRotationVectors.at(0) << endl;
	//cout << "index 0 --- Projector Translation:\n" << projTranslationVectors.at(0) << endl;
	///*カメラ*/
	//cout << "*********Camera Parameters*********" << endl;
	//cout << "Camera Matrix:\n" << cameraMatrix << endl;
	//cout << "Camera DistCoeffs:\n" << cameraDistCoeffs << endl;
	//cout << "index 0 --- Camera Rotate:\n" << cameraRotationVectors.at(0) << endl;
	//cout << "index 0 --- Camera Translation:\n" << cameraTranslationVectors.at(0) << endl;

	waitKey(0);

	return 0;
}

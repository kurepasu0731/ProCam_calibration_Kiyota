#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

//�L�����u���[�V�����֌W�ϐ�
#define IMAGE_NUM  (10)         /* �摜�� */
#define PAT_ROW    (7)          /* �p�^�[���̍s�� */
#define PAT_COL    (10)         /* �p�^�[���̗� */
#define PAT_SIZE   (PAT_ROW*PAT_COL)
#define ALL_POINTS (IMAGE_NUM*PAT_SIZE)
#define CHESS_SIZE (24.0)       /* �p�^�[��1�}�X��1�ӃT�C�Y[mm] */

#define PROJECTOR_WIDTH (1280) //�v���W�F�N�^�𑜓x
#define PROJECTOR_HEIGHT (800) //�v���W�F�N�^�𑜓x
//1280 * 800�̏ꍇ
#define PROJECTOR_CHESS_SIZE_WIDTH (103) /* �v���W�F�N�^�摜��ł̃p�^�[��1�}�X�̕��T�C�Y[pixel] */
#define PROJECTOR_CHESS_SIZE_HEIGHT (90) /* �v���W�F�N�^�摜��ł̃p�^�[��1�}�X�̍����T�C�Y[pixel] */ //�ڑ��l


const Size patternSize(10, 7); //�`�F�b�J�p�^�[���̌�_�̐�

const char* windowname_proj = "Projector Pattern";
const char* windowname_cam = "Camera Pattern";

//�e�햼�O
const string	windowNameUnd_R = "Undistorted RPlane Image";
const string	windowNameUnd_B = "Undistorted BPlane Image";
const string	windowNameUnd_src = "Undistorted src Image";

const string srcfoldername = "../src_image_0424/";
const string rPlanefoldername = "../Rplane_image_0424/";
const string bPlanefoldername = "../Bplane_image_0424/";
const string undistortedfoldername = "../undistorted_image_0424/";

//���͉摜�z��
vector<Mat> checkerimg_src;

//�J�����p�`�F�b�J�p�^�[���摜�z��
vector<Mat> checkerimg_cam;
//�J�����p�`�F�b�J�p�^�[���摜�z��(�J�����̘c�ݕ␳��)
vector<Mat> checkerimg_cam_undist;

//�v���W�F�N�^�p�`�F�b�J�p�^�[���摜�z��
vector<Mat> checkerimg_proj;
//�v���W�F�N�^�p�`�F�b�J�p�^�[���摜�z��(�J�����̘c�ݕ␳��)
vector<Mat> checkerimg_proj_undistCamera;
//�v���W�F�N�^�p�`�F�b�J�p�^�[���摜�z��(�J�����ƃv���W�F�N�^�̘c�ݕ␳��)
vector<Mat> checkerimg_proj_undistCameraProj;

//�J�����p�`�F�b�J�p�^�[���̍�������_�Ƃ���A�`�F�b�J��_�̐��E���W
vector<vector<Point3f>> worldPoints(IMAGE_NUM);

//�J�����L�����u���[�V�����p �J�����摜���W�ł̃`�F�b�J��_���W
vector<vector<Point2f>> cameraImagePoints_camera(IMAGE_NUM);
//�v���W�F�N�^�L�����u���[�V�����p �J�����摜���W�ł̃`�F�b�J��_���W
vector<vector<Point2f>> cameraImagePoints_proj(IMAGE_NUM);
//�v���W�F�N�^�L�����u���[�V�����p �v���W�F�N�^�摜��ł̃`�F�b�J�{�[�h�̌�_���W
vector<vector<Point2f>> projImagePoints_checkerboard;


// �Ή����郏�[���h���W�n�p�^�[��
TermCriteria criteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.001 );  

/*�J�����p�����[�^�s��*/
cv::Mat				cameraMatrix;		// �����p�����[�^�s��
cv::Mat				cameraDistCoeffs;		// �����Y�c�ݍs��
cv::vector<cv::Mat>	cameraRotationVectors;	// �B�e�摜���Ƃɓ������]�x�N�g��
cv::vector<cv::Mat>	cameraTranslationVectors;	// �B�e�摜���Ƃɓ����镽�s�ړ��x�N�g��

/*�v���W�F�N�^�p�����[�^�s��*/
cv::Mat				projMatrix;		// �����p�����[�^�s��
cv::Mat				projDistCoeffs;		// �����Y�c�ݍs��
cv::vector<cv::Mat>	projRotationVectors;	// �B�e�摜���Ƃɓ������]�x�N�g��
cv::vector<cv::Mat>	projTranslationVectors;	// �B�e�摜���Ƃɓ����镽�s�ړ��x�N�g��


Mat bin_b, bin_r; //R,B�v���[���̌���


//�t�H���_���Ɖ摜�ԍ�����t�@�C�����𐶐�
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

	//IplImage�֕ϊ�
	IplImage ipl = src;

	int roi_w = (int)(src.cols / dev);
	int roi_h = (int)(src.rows / dev);

	for(int y = 0; y < src.rows; y+=roi_h){
		for(int x = 0; x < src.cols; x+= roi_w){
			//ROI���Z�b�g
			cvSetImageROI(&ipl, Rect(x, y, roi_w,roi_h));

			//��l��
			cvThreshold(&ipl, &ipl, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
			//ROI����
			cvResetImageROI(&ipl);
		}
	}

	//Mat�֕ϊ�
	Mat result = srcimg.clone();
    result = cv::cvarrToMat(&ipl);  // �f�[�^���R�s�[����
	if (flag) bin_r = result.clone(); //1:b 0:r
	else bin_b = result.clone();
  //CV_Assert(reinterpret_cast<uchar*>(ipl.imageData) != result.data);

	waitKey(0);

}

void RPlanefilter(){
	Mat dst_r, gray_r;
	for(int i = 0; i < IMAGE_NUM; i++){
		//�F����
		Mat r_image = getRedImage(checkerimg_src[i]);
		//������(�o�C���e�����t�B���^)
		cv::bilateralFilter(r_image, dst_r, 11, 40, 200);
		//�O���[�X�P�[��
		cvtColor(dst_r, gray_r, CV_BGR2GRAY);
		//��l��
		threshold_div(gray_r, 6, 1);
		//cv::threshold(gray_r, bin_r, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
		//�ۑ�
		imwrite(get_capImgFileName(rPlanefoldername, i), bin_r);
	}
}

void BPlanefilter(){
	Mat dst_b,  gray_b;
	for(int i = 0; i < IMAGE_NUM; i++){
		//�F����
		Mat r_image = getBlueImage(checkerimg_src[i]);
		//������(�o�C���e�����t�B���^)
		cv::bilateralFilter(r_image, dst_b, 11, 40, 200);
		//�O���[�X�P�[��
		cvtColor(dst_b, gray_b, CV_BGR2GRAY);
		//��l��
		//b�̂�6�����ɂ��ē�l��
		threshold_div(gray_b, 6, 0);
		//cv::threshold(gray_b, bin_b, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

		//���]
		//Mat rev = ~bin_b;
		//�ۑ�
		//imwrite(get_capImgFileName(bPlanefoldername, i), rev);
		imwrite(get_capImgFileName(bPlanefoldername, i), bin_b);
	}
}

void findProjectorChessboardCorners(){
	//�\���p�E�B���h�E
	cv::namedWindow( windowname_proj, CV_WINDOW_AUTOSIZE );

	// �`�F�b�N�p�^�[���̌�_���W�����߁CimagePoints�Ɋi�[����
	for(int i = 0; i < IMAGE_NUM; i++){
		cout << "Find corners from image " << i;
		bool result = findChessboardCorners(checkerimg_proj_undistCamera[i], patternSize, cameraImagePoints_proj[i]);
		//bool result = findChessboardCorners(checkerimg_proj[i], patternSize, cameraImagePoints_proj[i]);
		if(result){
			cout << " ... All corners found." << endl;
			cout << cameraImagePoints_proj[i].size() << endl;
			//cv::Mat grayImg;
			//cv::cvtColor(checkerimg_proj_undistCamera[i], grayImg, CV_BGR2GRAY);	//�O���[�X�P�[����
			cv::cornerSubPix(checkerimg_proj_undistCamera[i], cameraImagePoints_proj[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //�R�[�i�[�ʒu���T�u�s�N�Z�����x�ɏC��
			//cv::cornerSubPix(checkerimg_proj[i], cameraImagePoints_proj[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //�R�[�i�[�ʒu���T�u�s�N�Z�����x�ɏC��
			// ���o�_��`�悷��
			//cv::drawChessboardCorners( checkerimg_proj_undistCamera[i], patternSize, ( cv::Mat )( cameraImagePoints_proj[i] ), true );

			//�\��
			cv::imshow( windowname_proj, checkerimg_proj_undistCamera[i] );
			//cv::imshow( windowname_proj, checkerimg_proj[i] );
			//�ۑ�
			imwrite(get_capImgFileName(bPlanefoldername, i), checkerimg_proj_undistCamera[i]);
			//imwrite(get_capImgFileName(bPlanefoldername, i), checkerimg_proj[i]);
			waitKey( 0 );
		}else{
			cout << " ... at least 1 corner not found." << endl;
			cout << cameraImagePoints_proj[i].size() << endl;
			waitKey( 1 );
		}
	}
}

void CameraCalibration(){
	//�\���p�E�B���h�E
	cv::namedWindow( windowname_cam, CV_WINDOW_AUTOSIZE );

	// �`�F�b�N�p�^�[���̌�_���W�����߁CimagePoints�Ɋi�[����
	for(int i = 0; i < IMAGE_NUM; i++){
		cout << "Find corners from image " << i;
		bool result = findChessboardCorners(checkerimg_cam[i], patternSize, cameraImagePoints_camera[i]);
		if(result){
			cout << " ... All corners found." << endl;
			cout << cameraImagePoints_camera[i].size() << endl;
			//cv::Mat grayImg;
			//cv::cvtColor(checkerimg_cam[i], grayImg, CV_BGR2GRAY);	//�O���[�X�P�[����
			cv::cornerSubPix(checkerimg_cam[i], cameraImagePoints_camera[i], cv::Size(3, 3), cv::Size(-1, -1), criteria);  //�R�[�i�[�ʒu���T�u�s�N�Z�����x�ɏC��
			// ���o�_��`�悷��
			//cv::drawChessboardCorners( checkerimg_cam[i], patternSize, ( cv::Mat )( cameraImagePoints_camera[i] ), true );
			//�\��
			cv::imshow( windowname_cam, checkerimg_cam[i] );
			waitKey( 0 );
		}else{
			cout << " ... at least 1 corner not found." << endl;
			cout << cameraImagePoints_camera[i].size() << endl;
			waitKey( 1 );
		}
	}

//�J�����p�`�F�b�J�p�^�[���̍�������_�Ƃ���A�`�F�b�J��_�̐��E���W�����߂�
	for( int i = 0; i < IMAGE_NUM; i++ ) {
		for( int j = 0 ; j < patternSize.area(); j++ ) {
			worldPoints[i].push_back( cv::Point3f(	static_cast<float>( j % patternSize.width *  CHESS_SIZE), 
				static_cast<float>( j / patternSize.width * CHESS_SIZE ), 
						  0.0 ) );
		}
	}

	// ����܂ł̒l���g���ăL�����u���[�V����
	cv::calibrateCamera( worldPoints, cameraImagePoints_camera, checkerimg_cam[0].size(), cameraMatrix, cameraDistCoeffs, 
			     cameraRotationVectors, cameraTranslationVectors );
	cout << "Camera parameters have been estimated" << endl << endl;


	//Kinect�̉摜�͘c�ݕ␳�ς݁H�H
	//R�v���[���摜��BPlane�摜����J�����̘c�݂�␳���A�␳��̉摜��ۑ�
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
		//�ۑ�
		imwrite(get_capImgFileName(rPlanefoldername, i), undistorted_r);
		imwrite(get_capImgFileName(bPlanefoldername, i), undistorted_b);
		imwrite(get_capImgFileName(undistortedfoldername, i), undistorted_src);


		waitKey( 0 );
	}
}

Point2f multHomography(Mat h, Point2f p)
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
	//(1) �摜�̃��[�h:src
	for(int i = 0; i < IMAGE_NUM; i++){
		checkerimg_src.push_back(imread(get_capImgFileName(srcfoldername, i)));
	}

	//(2) R,B�v���[���ŕ�������l�����ۑ�
	RPlanefilter();
	BPlanefilter();

	//�摜�̃��[�h:rplane bplane
	for(int i = 0; i < IMAGE_NUM; i++){
		//�O���[�X�P�[���œǂݍ��� imread-> flag: ��:3channel 0:grayscale ��:���̂܂�
		checkerimg_cam.push_back(imread(get_capImgFileName(rPlanefoldername, i), 0));
		checkerimg_proj.push_back(imread(get_capImgFileName(bPlanefoldername, i), 0));
	}

	//(3) �J�����̃p�����[�^����y�уJ�����̘c�ݕ␳
	CameraCalibration();

	//(4) �J�����E�v���W�F�N�^�Ԃ̃z���O���t�B�s��̐���(�`�F�b�J�p�^�[���̎l������)
	//�l���̃C���f�b�N�X
	int index0 = 0;
	int index1 = patternSize.width - 1;
	int index2 = patternSize.width * (patternSize.height - 1);
	int index3 = PAT_SIZE -1;

	//�摜(�J�����̘c�ݕ␳��)�̃��[�h:undistorted_r undistorted_b
	for(int i = 0; i < IMAGE_NUM; i++){
		checkerimg_cam_undist.push_back(imread(get_capImgFileName(rPlanefoldername, i), 0));
		checkerimg_proj_undistCamera.push_back(imread(get_capImgFileName(bPlanefoldername, i), 0));
	}

	//���x��B�v���[���摜����R�[�i�[���o
	findProjectorChessboardCorners();

	//�e�摜�̎l�����g���ď����z���O���t�B�s��H�𐄒�
	//��
	//(5) �J�����ŎB�e�����`�F�b�J�{�[�h�̌�_���z���O���t�B�ϊ����A�v���W�F�N�^�摜���W�ɕϊ�

	vector<vector<Point2f>> initial_proj_corners;

	//�v���W�F�N�^�摜��̓_(dst)
	Point2f corner0_projImg(172, 129);
	Point2f corner1_projImg(1102, 129);
	Point2f corner2_projImg(172, 677);
	Point2f corner3_projImg(1102, 677);

	for(int i = 0; i < IMAGE_NUM; i++)
	{

		//�J�����摜��Ō��o���ꂽ�_(src)
		Point2f corner0_detect = cameraImagePoints_proj[i][index0];
		Point2f corner1_detect = cameraImagePoints_proj[i][index1];
		Point2f corner2_detect = cameraImagePoints_proj[i][index2];
		Point2f corner3_detect = cameraImagePoints_proj[i][index3];

		//�J�����摜���v���W�F�N�^�摜�FH�@�Ƃ���
		Point2f src_pt[] = {corner0_detect, corner1_detect, corner2_detect, corner3_detect};
		Point2f dst_pt[] = {corner0_projImg, corner1_projImg, corner2_projImg, corner3_projImg};

		//�J�����摜���v���W�F�N�^�摜�ւ̃z���O���t�B�s��
		Mat H = getPerspectiveTransform(src_pt,dst_pt);

		cout << "\nH_" << i << ": \n" << H << endl;

		//�J�����ŎB�e�����`�F�b�J�{�[�h�̌�_���z���O���t�B�ϊ����A�v���W�F�N�^�摜���W�ɕϊ�
		vector<Point2f> checkerBoard_coners;
		for(int j = 0; j <  PAT_SIZE; j++)
		{
			Point2f dst = multHomography(H, cameraImagePoints_camera[i][j]);
			checkerBoard_coners.push_back(dst);

			cout << "(" << cameraImagePoints_camera[i][j].x << ", " << cameraImagePoints_camera[i][j].y << ") ---> "
				<< "(" << dst.x << ", " << dst.y << ")" << endl;
		}
		 projImagePoints_checkerboard.push_back(checkerBoard_coners);
		 checkerBoard_coners.clear();

	}

	//(6) �v���W�F�N�^�L�����u���[�V�����̎��s worldPoints --- projImagePoints_checkerboard
	cv::calibrateCamera( worldPoints, projImagePoints_checkerboard, Size(PROJECTOR_WIDTH, PROJECTOR_HEIGHT), projMatrix, projDistCoeffs, 
			     projRotationVectors, projTranslationVectors );

	cout << "Projector parameters have been estimated" << endl;

	//�o��
	/*�v���W�F�N�^*/
	cout << "*********Projector Parameters*********" << endl;
	cout << "Projector Matrix:\n" << projMatrix << endl;
	cout << "Projector DistCoeffs:\n" << projDistCoeffs << endl;
	cout << "index 0 --- Projector Rotate:\n" << projRotationVectors.at(0) << endl;
	cout << "index 0 --- Projector Translation:\n" << projTranslationVectors.at(0) << endl;
	/*�J����*/
	cout << "*********Camera Parameters*********" << endl;
	cout << "Camera Matrix:\n" << cameraMatrix << endl;
	cout << "Camera DistCoeffs:\n" << cameraDistCoeffs << endl;
	cout << "index 0 --- Camera Rotate:\n" << cameraRotationVectors.at(0) << endl;
	cout << "index 0 --- Camera Translation:\n" << cameraTranslationVectors.at(0) << endl;

	waitKey(0);

	return 0;
}

'''
InstagramML - Label Prediction Model
Author: Bill Dusch (bill.dusch@gmail.com)
'''

# Imports
import numpy as np
import pandas as pd
import shelve
import requests as req
from datetime import datetime
import time

# From Instagram ML
from api import API
from data import usernames, thresholds


label_models_file = '../src/models/labels.pkl'
label_columns_file = '../src/models/labelcolumns.pkl'
# Test data, actually gotten from request
"""
raw_file = {u'accounts': [{u'username': u'beautifuldestinations', u'updated': 1491060542, u'posts': [], u'id': u'276192188'},
                          {u'username': u'etdieucrea', u'updated': 1491060542, u'posts': [{u'updated': 1491060542, u'instagram': {u'code': u'BSWPjX2lsNl', u'dimensions': {u'width': 750, u'height': 500}, u'caption': u'Une photo pour quand maman travaille', u'comments_disabled': False, u'__typename': u'GraphImage', u'comments': {u'count': 1}, u'date': 1491060032, u'likes': {u'count': 167}, u'owner': {u'id': u'356564939'}, u'thumbnail_src': u'https://scontent-sjc2-1.cdninstagram.com/t51.2885-15/s640x640/sh0.08/e35/c166.0.667.667/17596200_1706155656348229_3289291370677665792_n.jpg', u'is_video': False, u'id': u'1483441528756421477', u'display_src': u'https://scontent-sjc2-1.cdninstagram.com/t51.2885-15/s750x750/sh0.08/e35/17596200_1706155656348229_3289291370677665792_n.jpg'}, u'annotations': {u'safeSearchAnnotation': {u'medical': u'VERY_UNLIKELY', u'spoof': u'VERY_UNLIKELY', u'violence': u'VERY_UNLIKELY', u'adult': u'VERY_UNLIKELY'}, u'imagePropertiesAnnotation': {u'dominantColors': {u'colors': [{u'color': {u'blue': 16, u'green': 19, u'red': 12}, u'pixelFraction': 0.09755546, u'score': 0.053716853}, {u'color': {u'blue': 26, u'green': 56, u'red': 31}, u'pixelFraction': 0.03221669, u'score': 0.049608365}, {u'color': {u'blue': 128, u'green': 204, u'red': 188}, u'pixelFraction': 0.022181982, u'score': 0.044818133}, {u'color': {u'blue': 128, u'green': 154, u'red': 175}, u'pixelFraction': 0.045797493, u'score': 0.03927099}, {u'color': {u'blue': 54, u'green': 83, u'red': 91}, u'pixelFraction': 0.033876564, u'score': 0.039133735}, {u'color': {u'blue': 71, u'green': 25, u'red': 6}, u'pixelFraction': 0.0051305266, u'score': 0.007408016}, {u'color': {u'blue': 23, u'green': 59, u'red': 139}, u'pixelFraction': 0.0011317338, u'score': 0.0014694642}, {u'color': {u'blue': 94, u'green': 117, u'red': 136}, u'pixelFraction': 0.044514865, u'score': 0.03524132}, {u'color': {u'blue': 93, u'green': 164, u'red': 148}, u'pixelFraction': 0.018711332, u'score': 0.03258325}, {u'color': {u'blue': 162, u'green': 189, u'red': 210}, u'pixelFraction': 0.03772446, u'score': 0.032555345}]}}, u'webDetection': {u'webEntities': [{u'entityId': u'/m/01bgsw', u'score': 1.0732733, u'description': u'Toddler'}, {u'entityId': u'/m/068hy', u'score': 0.8548201, u'description': u'Pet'}], u'visuallySimilarImages': [{u'url': u'https://s-media-cache-ak0.pinimg.com/236x/77/49/18/77491869feb24ec6e5daec9fbb2fbeba.jpg'}, {u'url': u'https://thumb9.shutterstock.com/display_pic_with_logo/2391818/340340276/stock-photo-blond-baby-girl-in-blue-jeans-costume-walking-forward-or-standing-on-green-path-in-a-garden-or-340340276.jpg'}, {u'url': u'https://thumb7.shutterstock.com/display_pic_with_logo/2418950/527771731/stock-photo-people-with-down-syndrome-are-equally-happy-walking-in-nature-527771731.jpg'}, {u'url': u'https://thumb9.shutterstock.com/display_pic_with_logo/4503292/480420292/stock-photo--toddler-boy-having-fun-outside-in-the-park-cute-happy-boy-child-outdoors-480420292.jpg'}, {u'url': u'https://thumb1.shutterstock.com/display_pic_with_logo/1477532/466305830/stock-photo-children-play-outdoors-on-a-sunny-summer-day-466305830.jpg'}, {u'url': u'https://www.colourbox.com/preview/1236089-grandfather-walking-with-son-and-grandson-along-woodland-path.jpg'}, {u'url': u'https://s-media-cache-ak0.pinimg.com/736x/84/69/d5/8469d5390ddeea2ce94ef226786e6b47.jpg'}, {u'url': u'http://www.greenchildmagazine.com/wp-content/uploads/2017/03/Roughhousing-320x320.jpg'}, {u'url': u'https://thumb7.shutterstock.com/display_pic_with_logo/64260/321934835/stock-photo-family-pet-animal-technology-and-people-concept-happy-family-with-labrador-retriever-dog-321934835.jpg'}, {u'url': u'http://previews.123rf.com/images/stockbroker/stockbroker1507/stockbroker150705375/42270941-Senior-Couple-Walking-In-Summer-Countryside-Stock-Photo-active.jpg'}, {u'url': u'https://thumb9.shutterstock.com/display_pic_with_logo/187633/289559282/stock-photo-senior-couple-walking-in-summer-countryside-289559282.jpg'}, {u'url': u'http://media.gettyimages.com/photos/couple-walking-with-grandson-on-rural-road-picture-id164853382?s=170667a'}, {u'url': u'https://thumb1.shutterstock.com/display_pic_with_logo/84928/515800930/stock-photo-happy-young-woman-with-long-curly-hairs-walking-on-city-street-fashion-model-in-ripped-jeans-515800930.jpg'}, {u'url': u'https://s-media-cache-ak0.pinimg.com/736x/7c/5f/73/7c5f733a62a8106477bf267ad44712ab.jpg'}, {u'url': u'https://thumb1.shutterstock.com/display_pic_with_logo/530809/278022398/stock-photo-young-mother-adjusting-son-s-shirt-collar-while-walking-in-a-park-278022398.jpg'}, {u'url': u'http://media.gettyimages.com/photos/happy-family-walking-on-trail-through-autumn-woods-picture-id182507749?s=170667a'}, {u'url': u'https://image.shutterstock.com/z/stock-photo-young-mother-and-her-cute-little-son-walking-in-a-park-288718547.jpg'}, {u'url': u'https://thumb1.shutterstock.com/display_pic_with_logo/438058/102961970/stock-photo-one-year-boy-walking-in-the-park-portrait-102961970.jpg'}, {u'url': u'https://s-media-cache-ak0.pinimg.com/236x/2a/09/fe/2a09fec2b5f74c410f11a7ae0b98ea79.jpg'}, {u'url': u'https://thumb10.shutterstock.com/display_pic_with_logo/452452/520449283/stock-photo-girl-play-with-kitten-520449283.jpg'}]}, u'labelAnnotations': [{u'score': 0.775443, u'mid': u'/m/06wtgq', u'description': u'play'}, {u'score': 0.75321084, u'mid': u'/m/0ytgt', u'description': u'child'}, {u'score': 0.7175661, u'mid': u'/m/01bgsw', u'description': u'toddler'}, {u'score': 0.6094571, u'mid': u'/m/02w8p8p', u'description': u'dog walking'}, {u'score': 0.5705285, u'mid': u'/m/083mg', u'description': u'walking'}], u'faceAnnotations': [{u'headwearLikelihood': u'VERY_UNLIKELY', u'panAngle': -4.27055, u'underExposedLikelihood': u'VERY_UNLIKELY', u'landmarkingConfidence': 0.6325829, u'detectionConfidence': 0.91786146, u'joyLikelihood': u'POSSIBLE', u'landmarks': [{u'position': {u'y': 89.5591, u'x': 484.76022, u'z': 0.00049717753}, u'type': u'LEFT_EYE'}, {u'position': {u'y': 88.170334, u'x': 513.67584, u'z': -2.1518097}, u'type': u'RIGHT_EYE'}, {u'position': {u'y': 82.41941, u'x': 476.1173, u'z': 2.9167821}, u'type': u'LEFT_OF_LEFT_EYEBROW'}, {u'position': {u'y': 82.10575, u'x': 491.93994, u'z': -6.337974}, u'type': u'RIGHT_OF_LEFT_EYEBROW'}, {u'position': {u'y': 81.56975, u'x': 505.4595, u'z': -7.345154}, u'type': u'LEFT_OF_RIGHT_EYEBROW'}, {u'position': {u'y': 80.69942, u'x': 522.50037, u'z': -0.55706906}, u'type': u'RIGHT_OF_RIGHT_EYEBROW'}, {u'position': {u'y': 87.749275, u'x': 498.84167, u'z': -6.9997635}, u'type': u'MIDPOINT_BETWEEN_EYES'}, {u'position': {u'y': 102.65143, u'x': 498.7913, u'z': -14.316895}, u'type': u'NOSE_TIP'}, {u'position': {u'y': 113.14231, u'x': 499.4606, u'z': -7.5299187}, u'type': u'UPPER_LIP'}, {u'position': {u'y': 125.67627, u'x': 500.6426, u'z': -4.731846}, u'type': u'LOWER_LIP'}, {u'position': {u'y': 117.25229, u'x': 488.34824, u'z': 1.432959}, u'type': u'MOUTH_LEFT'}, {u'position': {u'y': 115.9737, u'x': 513.9756, u'z': 0.029763697}, u'type': u'MOUTH_RIGHT'}, {u'position': {u'y': 118.05372, u'x': 500.13025, u'z': -5.185536}, u'type': u'MOUTH_CENTER'}, {u'position': {u'y': 104.85421, u'x': 507.1571, u'z': -4.2412047}, u'type': u'NOSE_BOTTOM_RIGHT'}, {u'position': {u'y': 105.39557, u'x': 491.84995, u'z': -2.972565}, u'type': u'NOSE_BOTTOM_LEFT'}, {u'position': {u'y': 106.75837, u'x': 499.1395, u'z': -7.891906}, u'type': u'NOSE_BOTTOM_CENTER'}, {u'position': {u'y': 86.96811, u'x': 485.28586, u'z': -2.0692391}, u'type': u'LEFT_EYE_TOP_BOUNDARY'}, {u'position': {u'y': 89.16645, u'x': 490.7819, u'z': -0.35706267}, u'type': u'LEFT_EYE_RIGHT_CORNER'}, {u'position': {u'y': 91.25252, u'x': 485.1751, u'z': -0.20731743}, u'type': u'LEFT_EYE_BOTTOM_BOUNDARY'}, {u'position': {u'y': 89.15459, u'x': 479.95566, u'z': 2.9579818}, u'type': u'LEFT_EYE_LEFT_CORNER'}, {u'position': {u'y': 88.80794, u'x': 485.0223, u'z': -0.7889949}, u'type': u'LEFT_EYE_PUPIL'}, {u'position': {u'y': 85.928986, u'x': 513.06866, u'z': -4.1371274}, u'type': u'RIGHT_EYE_TOP_BOUNDARY'}, {u'position': {u'y': 87.78333, u'x': 519.7243, u'z': 0.14917332}, u'type': u'RIGHT_EYE_RIGHT_CORNER'}, {u'position': {u'y': 89.795555, u'x': 513.77734, u'z': -2.3019128}, u'type': u'RIGHT_EYE_BOTTOM_BOUNDARY'}, {u'position': {u'y': 88.06782, u'x': 508.25748, u'z': -1.6200576}, u'type': u'RIGHT_EYE_LEFT_CORNER'}, {u'position': {u'y': 87.74153, u'x': 513.5265, u'z': -2.9453702}, u'type': u'RIGHT_EYE_PUPIL'}, {u'position': {u'y': 78.89675, u'x': 483.6973, u'z': -3.7521186}, u'type': u'LEFT_EYEBROW_UPPER_MIDPOINT'}, {u'position': {u'y': 77.77498, u'x': 513.8131, u'z': -5.979382}, u'type': u'RIGHT_EYEBROW_UPPER_MIDPOINT'}, {u'position': {u'y': 101.23767, u'x': 470.84457, u'z': 36.326714}, u'type': u'LEFT_EAR_TRAGION'}, {u'position': {u'y': 98.402824, u'x': 534.96356, u'z': 32.11991}, u'type': u'RIGHT_EAR_TRAGION'}, {u'position': {u'y': 81.47996, u'x': 498.6264, u'z': -7.9407096}, u'type': u'FOREHEAD_GLABELLA'}, {u'position': {u'y': 135.81595, u'x': 500.90445, u'z': -0.5492529}, u'type': u'CHIN_GNATHION'}, {u'position': {u'y': 119.37465, u'x': 473.83508, u'z': 25.625683}, u'type': u'CHIN_LEFT_GONION'}, {u'position': {u'y': 117.39006, u'x': 530.8938, u'z': 21.337954}, u'type': u'CHIN_RIGHT_GONION'}], u'sorrowLikelihood': u'VERY_UNLIKELY', u'surpriseLikelihood': u'VERY_UNLIKELY', u'tiltAngle': -2.5691485, u'angerLikelihood': u'VERY_UNLIKELY', u'boundingPoly': {u'vertices': [{u'y': 37, u'x': 455}, {u'y': 37, u'x': 548}, {u'y': 145, u'x': 548}, {u'y': 145, u'x': 455}]}, u'rollAngle': -2.3191476, u'blurredLikelihood': u'VERY_UNLIKELY', u'fdBoundingPoly': {u'vertices': [{u'y': 66, u'x': 466}, {u'y': 66, u'x': 537}, {u'y': 138, u'x': 537}, {u'y': 138, u'x': 466}]}}], u'cropHintsAnnotation': {u'cropHints': [{u'confidence': 1, u'importanceFraction': 0.53999996, u'boundingPoly': {u'vertices': [{}, {u'x': 749}, {u'y': 499, u'x': 749}, {u'y': 499}]}}]}}}], u'id': u'356564939'},
                          {u'username': u'instagood', u'updated': 1491060541, u'posts': [], u'id': u'5136815'}, {u'username': u'josecabaco', u'updated': 1491060541, u'posts': [], u'id': u'364395'},
                          {u'username': u'kissinfashion', u'updated': 1491060542, u'posts': [], u'id': u'528405678'}],
            u'success': True}
"""
raw_file = {u'accounts': [{u'username': u'beautifuldestinations', u'updated': 1491068282, u'posts': [], u'id': u'276192188'}, {u'username': u'etdieucrea', u'updated': 1491068283, u'posts': [], u'id': u'356564939'}, {u'username': u'instagood', u'updated': 1491068281, u'posts': [], u'id': u'5136815'}, {u'username': u'josecabaco', u'updated': 1491068282, u'posts': [{u'updated': 1491068282, u'instagram': {u'code': u'BSWe21wjxdt', u'dimensions': {u'width': 1080, u'height': 1080}, u'caption': u'Still.', u'comments_disabled': False, u'__typename': u'GraphImage', u'comments': {u'count': 0}, u'date': 1491068056, u'likes': {u'count': 14}, u'owner': {u'id': u'364395'}, u'thumbnail_src': u'https://scontent-sjc2-1.cdninstagram.com/t51.2885-15/s640x640/sh0.08/e35/17662387_460301490981077_237404708128423936_n.jpg', u'is_video': False, u'id': u'1483508837235234669', u'display_src': u'https://scontent-sjc2-1.cdninstagram.com/t51.2885-15/e35/17662387_460301490981077_237404708128423936_n.jpg'}, u'annotations': {u'safeSearchAnnotation': {u'medical': u'VERY_UNLIKELY', u'spoof': u'VERY_UNLIKELY', u'violence': u'VERY_UNLIKELY', u'adult': u'VERY_UNLIKELY'}, u'cropHintsAnnotation': {u'cropHints': [{u'confidence': 0.79999995, u'importanceFraction': 1, u'boundingPoly': {u'vertices': [{}, {u'x': 1079}, {u'y': 1079, u'x': 1079}, {u'y': 1079}]}}]}, u'imagePropertiesAnnotation': {u'dominantColors': {u'colors': [{u'color': {u'blue': 84, u'green': 85, u'red': 83}, u'pixelFraction': 0.23897777, u'score': 0.24168687}, {u'color': {u'blue': 194, u'green': 194, u'red': 193}, u'pixelFraction': 0.21284445, u'score': 0.23125286}, {u'color': {u'blue': 116, u'green': 116, u'red': 115}, u'pixelFraction': 0.194, u'score': 0.19089666}, {u'color': {u'blue': 157, u'green': 157, u'red': 156}, u'pixelFraction': 0.19284445, u'score': 0.1772988}, {u'color': {u'blue': 53, u'green': 54, u'red': 51}, u'pixelFraction': 0.13146667, u'score': 0.10970962}, {u'color': {u'blue': 227, u'green': 226, u'red': 225}, u'pixelFraction': 0.016266666, u'score': 0.040628083}, {u'color': {u'blue': 31, u'green': 32, u'red': 29}, u'pixelFraction': 0.0136, u'score': 0.008527118}]}}, u'labelAnnotations': [{u'score': 0.741505, u'mid': u'/m/0h8ls87', u'description': u'automotive exterior'}, {u'score': 0.68099743, u'mid': u'/m/03_qhc', u'description': u'grille'}, {u'score': 0.6456033, u'mid': u'/m/031bff', u'description': u'window covering'}, {u'score': 0.6289597, u'mid': u'/m/03scnj', u'description': u'line'}, {u'score': 0.5352665, u'mid': u'/m/04t7l', u'description': u'metal'}, {u'score': 0.5171048, u'mid': u'/m/01l0y0', u'description': u'material'}], u'webDetection': {u'webEntities': [{u'entityId': u'/m/0d4v4', u'score': 0.91372, u'description': u'Window'}, {u'entityId': u'/m/031bff', u'score': 0.903793, u'description': u'Window covering'}, {u'entityId': u'/m/025rw19', u'score': 0.7343757, u'description': u'Iron'}, {u'entityId': u'/m/0z8x4gb', u'score': 0.70686, u'description': u'Grille'}, {u'entityId': u'/m/01l0y0', u'score': 0.4697414, u'description': u'Material'}], u'visuallySimilarImages': [{u'url': u'https://s-media-cache-ak0.pinimg.com/236x/12/ab/3e/12ab3ea28ab1cdcd596d86bfe0f53bc8.jpg'}, {u'url': u'https://s-media-cache-ak0.pinimg.com/564x/12/ab/3e/12ab3ea28ab1cdcd596d86bfe0f53bc8.jpg'}, {u'url': u'https://4.imimg.com/data4/SU/OJ/MY-2374877/iron-tmt-bar-500x500.jpg'}, {u'url': u'https://us.123rf.com/450wm/srckomkrit/srckomkrit1607/srckomkrit160707354/60233995-metallic-background-fragment-of-steel-door.jpg?ver=6'}, {u'url': u'https://us.123rf.com/450wm/saksoni/saksoni1607/saksoni160700450/59809789-a-corrugated-fence-of-silver-metal-sheets-with-screw-texture-of-metal-fence.jpg?ver=6'}, {u'url': u'http://i.istockimg.com/file_thumbview_approve/19404816/2/stock-photo-19404816-corrugated-iron-frame-background.jpg'}, {u'url': u'https://thumbs.dreamstime.com/x/corrugated-iron-texture-13922679.jpg'}, {u'url': u'http://i.istockimg.com/file_thumbview_approve/19404737/2/stock-photo-19404737-corrugated-iron-frame-background.jpg'}, {u'url': u'https://thumb1.shutterstock.com/display_pic_with_logo/1450271/590511740/stock-photo-iron-curtain-590511740.jpg'}, {u'url': u'https://thumb7.shutterstock.com/display_pic_with_logo/4406098/454281121/stock-photo-old-iron-plate-wall-454281121.jpg'}, {u'url': u'http://st2.depositphotos.com/3284401/5972/i/950/depositphotos_59724315-stock-photo-texture-of-steel-ventilation-grille.jpg'}, {u'url': u'https://c1.staticflickr.com/4/3435/3698407288_2e738c68c9_b.jpg'}, {u'url': u'https://previews.123rf.com/images/edlits/edlits1406/edlits140600079/29000886-texture-of-steel-ventilation-grille-on-the-wall-of-a-building-Stock-Photo.jpg'}, {u'url': u'http://us.123rf.com/450wm/zerbor/zerbor1602/zerbor160200049/53771987-corrugated-sheet-metal-facade.jpg?ver=6'}, {u'url': u'https://thumb9.shutterstock.com/display_pic_with_logo/1961579/365541056/stock-photo-grille-of-fan-coil-unit-of-air-conditioner-365541056.jpg'}, {u'url': u'https://thumbs.dreamstime.com/x/metallica-black-grille-detail-protective-fence-51326972.jpg'}, {u'url': u'https://thumbs.dreamstime.com/x/silver-gray-corrugated-iron-wall-vertical-format-horizontal-lines-symmetrical-riveting-43989598.jpg'}, {u'url': u'https://us.123rf.com/450wm/pattanwit/pattanwit1511/pattanwit151100123/49563613-iron-wire-fence-stainless-steel-metal-mesh.jpg?ver=6'}, {u'url': u'https://thumb9.shutterstock.com/display_pic_with_logo/1717576/318070532/stock-photo-background-detail-of-texture-metal-door-corrugated-iron-panelling-318070532.jpg'}]}}}], u'id': u'364395'}, {u'username': u'kissinfashion', u'updated': 1491068282, u'posts': [], u'id': u'528405678'}], u'success': True}


class LabelModel(API):
    """
    Label Model.
    """
    def __init__(self, *args, **kwargs):
        API.__init__(self, *args, **kwargs)


    def get_request(self, get=False):
        """
        Request posts from API and return the account output and success
        """
        if get:
            file = API.get_posts(self)
        else:
            file = raw_file
        raw_data = file['accounts']
        success = file['success']
        return raw_data, success


    def munge_data(self, raw_data):
        """
        Take the raw data (JSON data converted into python dictionary) and return numpy arrays
        of the features and a dataframe of the username to detect if there are no new posts.
        :param raw_data: 
        :return: 
            X: Numpy array holding data (posts x label features) held in a dictionary on a per user basis
            verify: Boolean flag indicating there is data held in a dictionary on a per user basis
        """
        # Prepare dataframes
        num_accounts = len(raw_data)
        column_shelf = shelve.open(label_columns_file)
        labels = {}
        columns = {}
        for user in usernames:
            first_columns = [u'likes', u'username', u'instagram_id']
            time_columns = [u'hour_0005', u'hour_0611', u'hour_1217', u'hour_1823', u'monday', u'tuesday', u'wednesday', u'thursday', u'friday', u'saturday', u'sunday']
            user_columns = first_columns + time_columns + column_shelf[user]
            labels[user] = pd.DataFrame(index=[0], columns=user_columns)
            # Put in memory and make str
            columns[user] = [x.encode('utf-8') for x in column_shelf[user]]
        # Iterate over posts
        x = 0
        for i in range(num_accounts):
            user = raw_data[i]['username'].encode('utf-8')
            posts = raw_data[i]['posts']
            if posts:
                for j in range(len(posts)):
                    # Set likes
                    labels[user].set_value(x, 'likes', posts[j]['instagram']['likes']['count'])
                    # Set username
                    labels[user].set_value(x, 'username', user)
                    # Set id
                    labels[user].set_value(x, 'instagram_id', posts[j]['instagram']['id'])
                    # get date
                    date = posts[j]['instagram']['date']
                    (hour_0005, hour_0611, hour_1217, hour_1823,
                     monday, tuesday, wednesday, thursday, friday, saturday, sunday) = self.make_time(date)
                    # set dates
                    labels[user].set_value(x, 'hour_0005', hour_0005)
                    labels[user].set_value(x, 'hour_0611', hour_0611)
                    labels[user].set_value(x, 'hour_1217', hour_1217)
                    labels[user].set_value(x, 'hour_1823', hour_1823)
                    labels[user].set_value(x, 'monday', monday)
                    labels[user].set_value(x, 'tuesday', tuesday)
                    labels[user].set_value(x, 'wednesday', wednesday)
                    labels[user].set_value(x, 'thursday', thursday)
                    labels[user].set_value(x, 'friday', friday)
                    labels[user].set_value(x, 'saturday', saturday)
                    labels[user].set_value(x, 'sunday', sunday)
                    annotations = posts[j]['annotations']
                    if 'labelAnnotations' in annotations:
                        # Sets each label as a new feature with its  value the label's score
                        for item in annotations['labelAnnotations']:
                            description = item['description'].encode('utf-8').replace(' ', '_').replace("'", "")
                            score = np.float(item['score'])
                            if description in columns[user]:
                                labels[user].set_value(x, description, score)
                    x += 1
            else:
                # No new posts in this account
                pass
        # Prepare for training.
        # We can reduce the dataframe since we have the labels in memory seperately
        X = {}
        verify = {}
        posts = {}
        for user in usernames:
            X[user] = labels[user].ix[:, 3:].fillna(0).values
            posts[user] = labels[user].loc[:, 'instagram_id'].values.astype(str)
            verification = str(labels[user].loc[0, 'username'])
            if verification == 'nan':
                verify[user] = False
            elif verification == user:
                verify[user] = True
            else:
                # This shouldn't occur (user should match username but just in case
                verify[user] = False
        return X, verify, posts


    def make_time(self, date):
        """
        From a Linux time, convert it to a set of binary variables
        containing the hour (within a range of 6 hours) or the
        day of the week.
        :param date: 
        :return: 
        """
        # Make hour and day
        hour = datetime.fromtimestamp(date).hour
        day = datetime.fromtimestamp(date).day
        # convert this to hour stuff
        hour_0005 = 1 if (hour <= 5) else 0
        hour_0611 = 1 if ((hour >= 6) and (hour <= 11)) else 0
        hour_1217 = 1 if ((hour >= 12) and (hour <= 17)) else 0
        hour_1823 = 1 if ((hour >= 18) and (hour <= 23)) else 0
        monday = 1 if day == 0 else 0
        tuesday = 1 if day == 1 else 0
        wednesday = 1 if day == 2 else 0
        thursday = 1 if day == 3 else 0
        friday = 1 if day == 4 else 0
        saturday = 1 if day == 5 else 0
        sunday = 1 if day == 6 else 0
        return hour_0005, hour_0611, hour_1217, hour_1823, monday, tuesday, wednesday, thursday, friday, saturday, sunday


    def make_predictions(self, X, verify):
        """
        Iterate over all users and verify that there are predictions to be made; otherwise, return NAN.
        :param X: Data to predict over (dictionary of numpy arrays) 
        :param verify: dataframe to verify that this dataset isn't blank
        :return: Dictionary containing predicted likes
        """
        likes = {}
        for user in usernames:
            if verify[user]:
                likes[user] = self.predict(X[user], user)
#                print user, likes[user]
            else:
                likes[user] = np.nan
        return likes


    def predict(self, X, user):
        """
        Make a prediction on the dataset
        :param user: Username
        :param X: Numpy array holding the training data
        :return: 
        """
        if user not in usernames:
            print '{} not a valid username; run with {} to make prediction not crash'.format(user, usernames[0])
            user = usernames[0]
        models = shelve.open(label_models_file)
        model = models[user]
        y = model.predict(X)
        models.close()
        return y


    def prepare_predictions(self, likes, posts, verify):
        """
        Prepare predictions for output
        :param success: 
        :param likes: 
        :return: 
        """
        # Create object holding submissions
        submissions = {}
        for user in usernames:
            if verify[user]:
                submissions[user] = []
                for i in range(len(likes[user])):
                    submissions[user].append({})
                    submissions[user][i]['post'] = str(posts[user][i])
                    submissions[user][i]['likes'] = int(np.round(likes[user][i], 0))
            else:
                submissions[user] = None
        # List of all requests
        requests = []
        for user in usernames:
            if submissions[user] is None:
                pass
            else:
                # Append submissions to request list
                requests = requests + submissions[user]
        return requests


    def submit(self, requests):
        """
        Submit predictions to the Instagram ML server
        """
        messages = []
        # check to see if we have any requests to submit!
        if requests:
            for i in range(len(requests)):
                payload = requests[i]
                r = req.post(self.BASE_URL + 'submissions', json=payload)
                message = r.json()
                messages.append(message)
        return messages

    def main(self, get=False, submit=False):
        """
        Main method to get data, munge data, make, prepare, and submit predictions
        :param get: Flag to see if we are gettig data from instagram server or doing an offline test
        :param submit: Flag to see if we are actually submitting data to instagram server
        :return: 
        """
        raw_data, success = self.get_request(get=get)
        if success:
            X, verify, posts = self.munge_data(raw_data)
            likes = self.make_predictions(X, verify)
            requests = self.prepare_predictions(likes, posts, verify)
            if submit:
                messages = self.submit(requests)
                print requests
                print messages
            else:
                print requests


def run_all_the_things(model):
    """
    Infinite loop to run the server.
    :param model: 
    :return: 
    """
    while True:
        model.main(get=True, submit=True)
        dt = 60 * 5 # 5 minutes
        time.sleep(dt)


if __name__ == '__main__':
    username = 'yuyajeremyong@gmail.com'
    key = '6g6wDQF5sceQtEZeUKJWYx0o6Uer4vGg'
    model = LabelModel(username, key)
    model.main(get=True, submit=True)
#    run_all_the_things(model)

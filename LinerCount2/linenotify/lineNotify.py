
class LineNotify:

    def _lineNotify(self,payload,file=None):
        import requests
        url = 'https://notify-api.line.me/api/notify'
        token = 'IXTb4dLOabv1NZqfhhOOZtZFLryvPO3kCwmfNjVewYQ'
        headers = {'Authorization':'Bearer '+token}
        return requests.post(url,headers=headers,data=payload,files=file)
        
    def lineNotifyMessage(self,msg):
        payload = {'message':msg}
        return self._lineNotify(payload)
        
    def lineNotifyFile(self,filename):
        file = {'imageFile':open(filename,'rb')}
        payload = {'message':'detection'}
        return self._lineNotify(payload,file)
        
    def lineNotifyPicture(self,url):
        payload = {'message':" ",'imageThumbnail':url,'imageFullsize':url}
        return self._lineNotify(payload)
        
    def lineNotifySticker(self,stickerID,stickerPackageID):
        payload = {'message':" ",'stickerPackageId':stickerPackageID,'stickerId':stickerID}
        return self._lineNotify(payload)



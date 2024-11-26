// https://givemethesocks.tistory.com/39 참고

const _cardNameList = document.querySelectorAll("b.name");
const _carImg = document.querySelectorAll("a.anchor > figure > img");

let cardListText = '';


for (let num = 0; num < _cardNameList.length+1; num++) {
    let cardName = _cardNameList[num+1] ? _cardNameList[num+1].innerText : "No card found";

    let cardImgSrc = _carImg[num] ? _carImg[num].src : "No image found";


    cardListText += cardName + "***" + cardImgSrc + "\n";
}

function saveFile(fileName, content) {
    var blob = new Blob([content], { type: 'text/plain' });
    var objURL = window.URL.createObjectURL(blob);
    
    if (window.__Xr_objURL_forCreatingFile__) {
        window.URL.revokeObjectURL(window.__Xr_objURL_forCreatingFile__);
    }
    window.__Xr_objURL_forCreatingFile__ = objURL;

    var a = document.createElement('a');
    a.download = fileName;
    a.href = objURL;
    a.click();
}

saveFile("NaverCardList.csv", cardListText);
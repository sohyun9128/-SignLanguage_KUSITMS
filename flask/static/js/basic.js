function post(){
		document.getElementById("forPost").innerHTML = "<iframe src=\"guide\" scrolling=\"no\" height=\"160\" width=\"650\" id=\"iframe\"></iframe>";
}

function pre_post(){
		var iframes = document.querySelectorAll('iframe');
		for( var i = 0; i < iframes.length; i++){
				iframes[i].parentNode.removeChild(iframes[i]);
		}
}

function output(){
		document.getElementById("output").innerHTML='<object type="text/html" data="output"></object>';
}


function reset(){
		document.getElementById("output").innerHTML='&nbsp결과가 출력되는 공간입니다.';
}

function color() {
	fetch("/color").then(function(get_hand_hist){
		console.log(get_hand_hist);
	}).catch(function(err) {
		console.log(err);
	});
}


function predict() {
	fetch("/predict").then(function(prediction){
		console.log(prediction);
	}).catch(function(err) {
		console.log(err);
	});
}

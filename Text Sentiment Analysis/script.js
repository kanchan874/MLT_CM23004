let positiveWords = {};
let negativeWords = {};
let neutralWords = {};

let totalPos = 0;
let totalNeg = 0;
let totalNeu = 0;

let chart;

fetch("data.txt")
.then(response => response.text())
.then(data => trainModel(data));

function trainModel(data){

let lines = data.split("\n");
lines.shift();

lines.forEach(line=>{

let parts = line.split(",");

if(parts.length<2) return;

let text = parts[0].toLowerCase();
let label = parts[1].trim().toLowerCase();

let words = text.split(" ");

words.forEach(word=>{

word = word.replace(/[^a-z]/g,"");

if(!word) return;

if(label==="positive"){
positiveWords[word]=(positiveWords[word]||0)+1;
totalPos++;
}

else if(label==="negative"){
negativeWords[word]=(negativeWords[word]||0)+1;
totalNeg++;
}

else if(label==="neutral"){
neutralWords[word]=(neutralWords[word]||0)+1;
totalNeu++;
}

});

});

document.getElementById("status").innerText=
"Model trained successfully from dataset ✅";

}

function predict(){

let text=document.getElementById("inputText").value.toLowerCase();

if(!text.trim()) return alert("Enter sentence first");

let words=text.split(" ");

let posScore=0;
let negScore=0;
let neuScore=0;

words.forEach(word=>{

word=word.replace(/[^a-z]/g,"");

if(positiveWords[word])
posScore+=positiveWords[word]/totalPos;

if(negativeWords[word])
negScore+=negativeWords[word]/totalNeg;

if(neutralWords[word])
neuScore+=neutralWords[word]/totalNeu;

});

let total=posScore+negScore+neuScore;

if(total===0){
posScore=negScore=neuScore=1;
total=3;
}

let posProb=(posScore/total*100).toFixed(2);
let negProb=(negScore/total*100).toFixed(2);
let neuProb=(neuScore/total*100).toFixed(2);

let result="NEUTRAL 😐";

if(posProb>negProb && posProb>neuProb)
result="POSITIVE 😊";

else if(negProb>posProb && negProb>neuProb)
result="NEGATIVE 😞";

document.getElementById("predictionResult").innerText=
"Prediction : "+result;

updateChart(posProb,negProb,neuProb);

}

function updateChart(pos,neg,neu){

let ctx=document.getElementById("probChart").getContext("2d");

if(chart) chart.destroy();

chart=new Chart(ctx,{
type:"bar",
data:{
labels:["Positive","Negative","Neutral"],
datasets:[{
label:"Probability %",
data:[pos,neg,neu],
backgroundColor:[
"#00c853",
"#d50000",
"#ffab00"
]
}]
},
options:{
responsive:true,
scales:{
y:{
beginAtZero:true,
max:100
}
}
}
});

}
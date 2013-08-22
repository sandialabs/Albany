var parentLinkIsOpener=0; // whether to use parent link/text as opener, overriding its own link
if(window.attachEvent)window.attachEvent('onload',cm);
else if(window.addEventListener)window.addEventListener('load',cm,false);
function cm(){
 cmId=0;
 if(!document.getElementsByTagName)return; // reject non-compliant browsers
 a=document.getElementsByTagName('UL');
 for(i=0;a[i];i++){
  if(a[i].getElementsByTagName('UL')){
   // a[i] is an object which has a collapsible list in it
   b=a[i].childNodes;
   for(j=0;b[j];j++){
    if(b[j].nodeName=='LI'){
     d=b[j].getElementsByTagName('ul');
     if(d.length){
      c=document.createElement('a');
      c.setAttribute('href','javascript:cmSwitch("cm'+(cmId)+'")');
      c.setAttribute('id','cm'+(cmId)+'A');
      if(parentLinkIsOpener){
       // we've chosen to use the parent link as the opener
       var e=b[j].innerHTML;
       e=e.replace(/\n/g,'');
       e=e.replace(/(<ul|<UL).*/,'');
       e=e.replace(/<[^>]*>/g,'');
       c.innerHTML=e;
       b[j].replaceChild(c,b[j].childNodes[0]);
      }else{
       // create a new [+] link as the opener
       c.style.display='inline';
       c.innerHTML='[+]';
       b[j].insertBefore(c,b[j].firstChild);
      }
      d[0].setAttribute('id','cm'+(cmId++));
      d[0].style.display='none';
     }
    }
   }
  }
 }
}

function cmSwitch(id){
 a=document.getElementById(id);
 a.style.display=(a.style.display=='block')?'none':'block';
 if(parentLinkIsOpener)return;
 a=document.getElementById(id+'A');
 a.innerHTML=(a.innerHTML=='[+]')?'[-]':'[+]';
}

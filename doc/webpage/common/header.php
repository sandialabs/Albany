<?php 

  $packageName = "";
  $packageSubtitle = "";
  $packageDirName = "";
  $packageTDirName = "";
  $includePublications = false;
  $includeRelease106Docs = false;
  $release106DocsPagePath = "";
  $includeRelease104Docs = false;
  $release104DocsPagePath = "";
  $includeRelease102Docs = false;
  $release102DocsPagePath = "";
  $includeRelease100Docs = false;
  $release100DocsPagePath = "";
  $includeRelease90Docs = false;
  $release90DocsPagePath = "";
  $includeRelease80Docs = false;
  $release80DocsPagePath = "";
  $includeRelease70Docs = false;
  $release70DocsPagePath = "";
  $includeRelease60Docs = false;
  $release60DocsPagePath = "";
  $includeRelease50Docs = false;
  $release50DocsPagePath = "";
  $includeRelease40Docs = false;
  $release40DocsPagePath = "";
  $includeDevelopmentDocs = false;
  $developmentDocsPagePath = "";
  $additionalMenuItems = array();
  $quickLinks = array();

  function setPackageName ($in_packageName) { 
    global $packageName; $packageName = $in_packageName; }   
    
  function setPackageSubtitle ($in_packageSubtitle) { 
    global $packageSubtitle; $packageSubtitle = $in_packageSubtitle; } 
       
  function setPackageDirName ($in_packageDirName) { 
    global $packageDirName; $packageDirName = $in_packageDirName; }   
    
  function setPackageTDirName ($in_packageTDirName) { 
    global $packageTDirName; $packageTDirName = $in_packageTDirName; }
    
  function setPackageMailName ($in_packageMailName) { 
    global $packageMailName; $packageMailName = $in_packageMailName; }   
    
  function setIncludePublications ($in_includePublications) { 
    global $includePublications; $includePublications= $in_includePublications; 
  }

  function setIncludeRelease106Docs ($in_includeRelease106Docs, $in_release106DocsPagePath="") {
    global $includeRelease106Docs; $includeRelease106Docs = $in_includeRelease106Docs;
    global $release106DocsPagePath; $release106DocsPagePath = $in_release106DocsPagePath; }

  function setIncludeRelease104Docs ($in_includeRelease104Docs, $in_release104DocsPagePath="") {
    global $includeRelease104Docs; $includeRelease104Docs = $in_includeRelease104Docs;
    global $release104DocsPagePath; $release104DocsPagePath = $in_release104DocsPagePath; }

  function setIncludeRelease102Docs ($in_includeRelease102Docs, $in_release102DocsPagePath="") {
    global $includeRelease102Docs; $includeRelease102Docs = $in_includeRelease102Docs;
    global $release102DocsPagePath; $release102DocsPagePath = $in_release102DocsPagePath; }

  function setIncludeRelease100Docs ($in_includeRelease100Docs, $in_release100DocsPagePath="") {
    global $includeRelease100Docs; $includeRelease100Docs = $in_includeRelease100Docs;
    global $release100DocsPagePath; $release100DocsPagePath = $in_release100DocsPagePath; }
    
  function setIncludeRelease90Docs ($in_includeRelease90Docs, $in_release90DocsPagePath="") { 
    global $includeRelease90Docs; $includeRelease90Docs = $in_includeRelease90Docs; 
    global $release90DocsPagePath; $release90DocsPagePath = $in_release90DocsPagePath; }  
      
  function setIncludeRelease80Docs ($in_includeRelease80Docs, $in_release80DocsPagePath="") { 
    global $includeRelease80Docs; $includeRelease80Docs = $in_includeRelease80Docs; 
    global $release80DocsPagePath; $release80DocsPagePath = $in_release80DocsPagePath; } 
    
  function setIncludeRelease70Docs ($in_includeRelease70Docs, $in_release70DocsPagePath="") { 
    global $includeRelease70Docs; $includeRelease70Docs = $in_includeRelease70Docs; 
    global $release70DocsPagePath; $release70DocsPagePath = $in_release70DocsPagePath; } 
    
  function setIncludeRelease60Docs ($in_includeRelease60Docs, $in_release60DocsPagePath="") { 
    global $includeRelease60Docs; $includeRelease60Docs = $in_includeRelease60Docs; 
    global $release60DocsPagePath; $release60DocsPagePath = $in_release60DocsPagePath; } 
    
  function setIncludeRelease50Docs ($in_includeRelease50Docs, $in_release50DocsPagePath="") { 
    global $includeRelease50Docs; $includeRelease50Docs = $in_includeRelease50Docs; 
    global $release50DocsPagePath; $release50DocsPagePath = $in_release50DocsPagePath; } 
    
  function setIncludeRelease40Docs ($in_includeRelease40Docs, $in_release40DocsPagePath="") { 
    global $includeRelease40Docs; $includeRelease40Docs = $in_includeRelease40Docs; 
    global $release40DocsPagePath; $release40DocsPagePath = $in_release40DocsPagePath; }  
     
  function setIncludeDevelopmentDocs ($in_includeDevelopmentDocs, $in_developmentDocsPagePath="") { 
    global $includeDevelopmentDocs; $includeDevelopmentDocs = $in_includeDevelopmentDocs; 
    global $developmentDocsPagePath; $developmentDocsPagePath = $in_developmentDocsPagePath; }  
        
  function setAdditionalMenuItems ($url, $label, $isHeader) { 
    global $additionalMenuItems; $additionalMenuItems[] = new MenuItem($url, $label, $isHeader); }  
      
  function setQuickLinks ($url, $label, $isHeader) { 
    global $quickLinks; $quickLinks[] = new MenuItem($url, $label, $isHeader); }  
    
  class MenuItem {
    var $url = "";
    var $label = "";
    var $isHeader = false;
    
    function MenuItem($url, $label, $isHeader) {
      $this->url = $url;
      $this->label = $label;
      $this->isHeader = $isHeader;
    }    
  } 
  
?>

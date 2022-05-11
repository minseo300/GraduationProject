<?php
     include('connection.php');

      $case=$_GET["CASE"];
      $id=$_GET["ID"];
    
      //mysqli_select_db($conn,"SmartMirror") or die('DB 선택 실패');

      $info_table="R_dandy_info";
      $json_result=array();
  
      $sql="SELECT * FROM $info_table WHERE temperature_section=5 and ID=1";
      $result=mysqli_query($conn,$sql);
      while($row=mysqli_fetch_array($result)){
              array_push($json_result,array("Category"=>$row['outer_color']));
      }
      echo json_encode(array("Look"=>$json_result),JSON_UNESCAPED_UNICODE);
  
?>

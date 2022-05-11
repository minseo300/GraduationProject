<?php
    include('connection.php');

      $case=$_GET["CASE"];
      $id=$_GET["ID"];
    
      if($case=="top")
      {
          $top_cate=$_GET["TOP_CATEGORY"];
          $top_print=$_GET["TOP_PRINT"];
          $sleeve=$_GET["TOP_SLEEVE"];
          $temp=$_GET["TEMP"];
          
          $sql="UPDATE U_top_info SET category='$top_cate',sleevelength='$sleeve',top_print='$top_print',temperature_section='$temp' where ID=$id";
      }
      else if($case=="bottom")
      {
          $bottom_cate=$_GET["BOTTOM_CATEGORY"];
          $length=$_GET["BOTTOM_LENGTH"];
          $fit=$_GET["BOTTOM_FIT"];
          $temp=$_GET["TEMP"];

          $sql="UPDATE U_bottom_info SET category='$bottom_cate',bottom_fit='$fit',category='$bottom_cate',temperature_section='$temp' where ID=$id";
      }
      else if($case=="outer")
      {
          $outer_cate=$_GET["OUTER_CATEGORY"];
          $temp=$_GET["TEMP"];
          
          $sql="UPDATE U_outer_info SET category='$outer_cate',temperature_section='$temp' where ID=$id";
      }
      mysqli_query($conn,$sql);

      if(mysqli_query($conn,$sql))
      {
          echo "update successfully<br/>";
      }
      else{
          echo "fail";
      }
?>

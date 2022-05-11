<?php
     $conn=mysqli_connect('54.180.67.155','minseo','minseopw'); # DB연결
      if(mysqli_connect_errno())
      {
          echo '[연결실패]: '.mysqli_connect_error();
      }
      else{
          //echo '[연결성공]';
      }
      mysqli_select_db($conn,"SmartMirror") or die('DB 선택 실패');
?>

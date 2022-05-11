<?php

    // 앱에 사용자 옷들 전달
    include('connection.php');

    
   
    $category=$_GET["CATEGORY"];
    //echo "category: ".$category;
    $table='U_'.$category.'_image';
    $json_result=array();
    if($category!="")
    {
        $sql="SELECT * from $table";
        $result=mysqli_query($conn,$sql);
        while($row=mysqli_fetch_array($result)){
            // echo "ID: ".$row['ID']."<br/>";
            // echo "Image: ".$row['image']."<br/>";
            // echo "Image: ".$row['category']."<br/>";
            array_push($json_result,array("ID"=>$row['ID'],"Image"=>$row['image']));
        }
        echo json_encode(array("User_items"=>$json_result),JSON_UNESCAPED_UNICODE);
    }
?>

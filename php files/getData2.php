<?php

    include('connection.php');

    
    // $temp=isset($_GET['temp'])?$_GET['temp']:'';
    // $styling=isset($_GET['styling'])?$_GET['styling']:'';
    $temp=$_GET["TEMP"];
    $styling=$_GET["STYLING"];
    $android = strpos($_SERVER['HTTP_USER_AGENT'], "Android");

    $info_table="R_".$styling."_info";
    //echo "info table: $info_table<br/>";
    $image_table="R_".$styling."_image";
    $category="styling";
    //echo "image table: $image_table<br/>";
    $json_result=array();

    if($temp!="")
    {
        //$sql="SELECT * FROM  $info_table as n JOIN $image_table as m ON n.ID=m.ID WHERE n.temperature_section= $temp and m.category=styling"; // 카테고리가 styling인것만
        // $sql="SELECT * FROM $info_table JOIN $image_table ON $info_table.ID=$image_table.ID WHERE temperature_section=$temp and category='styling'";
        $sql="SELECT * FROM $info_table JOIN $image_table ON $info_table.ID=$image_table.ID WHERE temperature_section=$temp";
        //echo "sql: $sql<br/>";
        $result=mysqli_query($conn,$sql);
        while($row=mysqli_fetch_array($result)){
            // echo "ID: ".$row['ID']."<br/>";
            // echo "Image: ".$row['image']."<br/>";
            // echo "Image: ".$row['category']."<br/>";
            array_push($json_result,array("Category"=>$row['category'],'ID'=>$row['ID'],"Image"=>$row['image'],"Temp"=>$row['temperature_section'],
            "outerLink"=>$row['outer_link'],"topLink"=>$row['top_link'],"bottomLink"=>$row['bottom_link'],"onepieceLink"=>$row['onepiece_link']));
        }
        echo json_encode(array("Look"=>$json_result),JSON_UNESCAPED_UNICODE);
    }
    else{
        echo ".";
    }
?>

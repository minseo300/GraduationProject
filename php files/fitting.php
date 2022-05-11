<?php
        // 무신사 가상피팅 테스트용
        include('connection.php');


        $x=$_GET["X"];
        echo "X: ".$x;
        $y=$_GET["Y"];
        echo "Y: ".$y;
        $w=$_GET["w"];
        echo "w: ".$w;
        $h=$_GET["h"];
        echo "h: ".$h;
        $lh=$_GET["lh"];
        echo "lh: ".$lh;

        //mysqli_select_db($conn,"SmartMirror") or die('DB 선택 실패');
        $sql="SELECT * FROM R_casual_image WHERE ID=16"; // 데이터 확인용 임의 쿼리
        // $sql="SELECT * FROM $image_table WHERE ID=$id";
        $result=mysqli_query($conn,$sql);
        // $count=0;
        print("<body bgcolor='black'>"); // 배경 색 변경
        while($row=mysqli_fetch_array($result)){
            
            if($row['category']=="top")
            {
                $top=$row['image'];
                //echo "Image url: ".$row['image']."<br/>";
                //$count=$count+0; // 아우터가 없는 경우
            }
            else if($row['category']=="bottom")
            {
                $bottom=$row['image'];
            }
            else if($row['category']=="outer")
            {
                $outer=$row['image'];
                //$count=$count+1; //아우터가 있는 경우
            }
            else if($row['category']=="onepiece")
            {
                $onepiece=$row['image'];
                //$count=$count+2; //원피스인 경우
            }
           else if($row['category']=="styling")
           {
               $all=$row['image'];
               echo "ALL Image url: ".$row['image']."<br/>";
           }
        }

        $myWidth='<script>document.write(window.innerWidth);</script>';


        $ux=$x+100;
        $w=$w*5;
        $cropping=$w/2;
        $h=$h*5;
        $ly=$y+$h;
        $lw=$w-40;
        $lh=$lh*5;
        $lx=$ux+8;

        //echo "<img src='".$top."'>";
        //echo "<img src='".$outer."'>";
        ///////////////////////////////////// cropping success
        //echo "<img src='".$top."' style='position: absolute; z-index:1; float:left; margin-left: ".$x."px; margin-top: ".$y."px; width: ".$w."px; height: ".$h."px;' >";
        //echo "<img src='".$all."' style='position:absolute; float:left; margin-left: 5px; margin-top:50px; width: 100px; height:200px;'>";
        echo "<img src='".$outer."' style='position:absolute; z-index:2; float:left; margin-left: ".$ux."px; margin-top: ".$y."px; clip: rect(0,".$cropping."px, ".$h."px, 0); height: ".$h."px; width: ".$w."px;' >";
        echo "<img src='".$top."' style='position:absolute; z-index:1; float:left; margin-left: ".$ux."px; margin-top: ".$y."px; width: ".$w."px; height: ".$h."px;' >";
        echo "<img src='".$bottom."' style='position:absolute; float:left; margin-left: ".$lx."px; margin-top: ".$ly."px; width: ".$lw."px; height: ".$lh."px;' >";
        
        //////////////////////////////////////////////////////////////////

        //echo "<img src='".$outer."' style='position: absolute; z-index:2; float:left; margin-left: ".$x."px; margin-top: ".$y."px; width: ".$w."px; height: ".$h."px; max-width: ".$cropping."px; overflow:hidden;' >";
        
        // echo "<div style='float:left; position:absolute; width: 1000px; height: 1000px; top: 500px; left: 500px;'>;
        //             <img src='".$top."' width= '".$w."' height='".$h."' ></div>";
        // echo "<div style='float:left; position:absolute; width: ".$w."px; height: ".$h."px; top:".$y."px; left: ".$x."px;'>;
        //             <img src='".$top."' width= '".$w."' height='".$h."' ></div>";
        // echo "<div style='float:left;'>
        //             <img src='".$bottom."' width='600' height='600'></div>";

    //     if($count==0) // 상의, 하의만 있는 경우
    //     {

    //         echo "<div style='float:left;'>
    //                 <img src='".$top."' width='600' height='600' ></div>";
    //         echo "<div style='float:left;'>
    //                 <img src='".$bottom."' width='600' height='600'></div>";
    //     }
    //     else if($count==1) //아우터가 있는 경우
    //     {
    //         echo "<div style='float:left;'>
    //                 <img src='".$outer."' width='600' height='600' ></div>";
    //         echo "<div style='float:left;'>
    //                 <img src='".$top."' width='600' height='600' ></div>";
    //         echo "<div style='float:left;'>
    //                 <img src='".$bottom."' width='600' height='600'></div>";
    //     }
    //     else if($count==2) // 원피스만 있는 경우
    //     {
    //         echo "<div style='float:left;'>
    //                 <img src='".$onepiece."' width='600' height='600' ></div>";
    //     }
    //     else if($count==3) // 아우터, 원피스인 경우
    //     {
    //         echo "<div style='float:left;'>
    //                 <img src='".$outer."' width='600' height='600' ></div>";
    //         echo "<div style='float:left;'>
    //                 <img src='".$onepiece."' width='600' height='600'></div>";   
    //     }
       
        




    //     // $id=$_GET["STYLING"];
    //     // $styling=$_GET["ID"];
    //     // $android = strpos($_SERVER['HTTP_USER_AGENT'], "Android");

    //     // $image_table="R_".$styling."_image";
    //     // //$category="styling";
    //     // //echo "image table: $image_table<br/>";
    //     // $json_result=array();


    //     // if($id!="")
    //     // {
    //     //     //$sql="SELECT * FROM $image_table WHERE ID=$id"; 
    //     //     $sql="SELECT * FROM R_casual_image WHERE ID=2"; 
    //     //     //echo "sql: $sql<br/>";
    //     //     $result=mysqli_query($conn,$sql);
    //     //     while($row=mysqli_fetch_array($result)){
                
    //     //        echo "result: ".$row;
    //     //     }
    //     //     //echo json_encode(array("Item"=>$json_result),JSON_UNESCAPED_UNICODE);
    //     // }
    //     // else{
    //     //     echo ".";
    //     // }

             
    // 
    ?>

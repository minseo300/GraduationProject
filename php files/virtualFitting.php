    <?php
        include('connection.php');

        $sql="SELECT * FROM R_formal_image WHERE ID=4"; //R_casual_image
        $result=mysqli_query($conn,$sql);
        print("<body bgcolor='black'>"); // 배경 색 변경
        while($row=mysqli_fetch_array($result)){
                
            //echo "ID: ".$row['ID']."<br/>";
            // echo "Image url: ".$row['image']."<br/>";
            if($row['category']=="top")
            {
                $top=$row['image'];
            }
            else if($row['category']=="bottom")
            {
                $bottom=$row['image'];
            }
            else if($row['category']=="outer")
            {
                $outer=$row['image'];
            }
            else if($row['category']=="onepiece")
            {
                $onepiece=$row['image'];
            }
           else if($row['category']=="styling")
           {
               $all=$row['category'];
               echo "Image url: ".$row['image']."<br/>";
           }

            //echo "Category: ".$row['category']."<br/>";
        }
        echo "<div style='float:left;'>
        <img src='".$top."' width='600' height='600' ></div>";
        echo "<div style='float:left;'>
        <img src='".$bottom."' width='600' height='600'></div>";




        // $id=$_GET["STYLING"];
        // $styling=$_GET["ID"];
        // $android = strpos($_SERVER['HTTP_USER_AGENT'], "Android");

        // $image_table="R_".$styling."_image";
        // //$category="styling";
        // //echo "image table: $image_table<br/>";
        // $json_result=array();


        // if($id!="")
        // {
        //     //$sql="SELECT * FROM $image_table WHERE ID=$id"; 
        //     $sql="SELECT * FROM R_casual_image WHERE ID=2"; 
        //     //echo "sql: $sql<br/>";
        //     $result=mysqli_query($conn,$sql);
        //     while($row=mysqli_fetch_array($result)){
                
        //        echo "result: ".$row;
        //     }
        //     //echo json_encode(array("Item"=>$json_result),JSON_UNESCAPED_UNICODE);
        // }
        // else{
        //     echo ".";
        // }

        
    ?>

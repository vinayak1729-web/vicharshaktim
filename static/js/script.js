function loadData(action) {
    $.ajax({
        url: '/gee_data',
        type: 'POST',
        data: { action: action },
        success: function(response) {
            if (response.error) {
                alert('Error: ' + response.error);
            } else {
                $('#map').html(response.map_html);
            }
        },
        error: function() {
            alert('Failed to load data');
        }
    });
}